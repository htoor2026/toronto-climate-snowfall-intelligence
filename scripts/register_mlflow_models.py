#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
except ImportError as exc:
    raise SystemExit('MLflow is not installed. Run: python -m pip install -U mlflow') from exc

from src.training.evaluate import FINAL_FEATURES
from src.tracking.mlflow_registry import (
    build_climatology_estimator,
    dataframe_hash,
    find_version_by_tags,
    load_model_comparison,
    set_version_tags,
    sha256_file,
    sqlite_tracking_uri,
)


def register_sklearn_version(*, client, registered_model_name, experiment_name, role, method, estimator, X_example, metrics, snapshot_hash, tags, artifact_paths):
    existing = find_version_by_tags(client, model_name=registered_model_name, role=role, snapshot_hash=snapshot_hash)
    if existing is not None:
        return existing, False
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f'{role}-{snapshot_hash[:8]}'):
        mlflow.set_tags({'project':'toronto-climate-analysis','task':'seasonal-snowfall-forecast','role':role,'method':method,'snapshot_hash':snapshot_hash})
        mlflow.log_params({'role':role,'method':method,'feature_count':len(FINAL_FEATURES)})
        mlflow.log_metrics(metrics)
        for artifact in artifact_paths:
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path='evaluation')
        predictions = estimator.predict(X_example)
        signature = infer_signature(X_example, predictions)
        model_info = mlflow.sklearn.log_model(
        name="model",
        sk_model=estimator,
        signature=signature,
        input_example=X_example,
        skops_trusted_types=[
            "sklearn.model_selection._split.TimeSeriesSplit"
        ],
    )
        model_version = mlflow.register_model(model_uri=model_info.model_uri, name=registered_model_name)
    set_version_tags(client, model_name=registered_model_name, version=model_version.version, tags={**tags,'role':role,'method':method,'snapshot_hash':snapshot_hash})
    return model_version, True


def main() -> int:
    config = json.loads((PROJECT_ROOT/'config'/'mlflow_config.json').read_text(encoding='utf-8'))
    tracking_db = PROJECT_ROOT / config['tracking_db']
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = sqlite_tracking_uri(tracking_db)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    model_name = config['registered_model_name']

    training_path = PROJECT_ROOT/'data'/'processed'/'production_training_dataset.csv'
    strict_history_path = PROJECT_ROOT/'data'/'processed'/'forecast_features_full.csv'
    metrics_path = PROJECT_ROOT/'data'/'processed'/'retraining_model_comparison.csv'
    candidate_path = PROJECT_ROOT/'models'/'retraining'/'elasticnet_nao_candidate.joblib'
    decision_path = PROJECT_ROOT/'models'/'retraining'/'latest_retraining_decision.json'
    report_path = PROJECT_ROOT/'reports'/'retraining'/'latest_retraining_report.md'
    required = [training_path, strict_history_path, metrics_path, candidate_path, decision_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Run scripts/retrain_model.py --force first. Missing: {missing}')

    training = pd.read_csv(training_path)
    strict_history = pd.read_csv(strict_history_path)
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    comparison = load_model_comparison(metrics_path)
    X_complete = training.dropna(subset=FINAL_FEATURES).sort_values('season_year')[FINAL_FEATURES]
    if X_complete.empty:
        raise ValueError('No model-ready feature rows found.')
    X_example = X_complete.tail(1)

    champion_model, champion_expected_cm, champion_history = build_climatology_estimator(strict_history, training, window=int(config['climatology_window']))
    challenger_model = joblib.load(candidate_path)

    # TimeSeriesSplit was needed during fitting, not inference.
    if hasattr(challenger_model, "named_steps"):
        fitted_model = challenger_model.named_steps.get("model")
        if fitted_model is not None and hasattr(fitted_model, "cv"):
            fitted_model.cv = None
    champion_snapshot_hash = dataframe_hash(champion_history)
    challenger_snapshot_hash = sha256_file(candidate_path)
    artifacts = [metrics_path, decision_path, report_path]

    champion_version, champion_created = register_sklearn_version(
        client=client, registered_model_name=model_name, experiment_name=config['experiment_name'],
        role='champion', method='Historical Climatology', estimator=champion_model, X_example=X_example,
        metrics=comparison['Historical Climatology'], snapshot_hash=champion_snapshot_hash,
        tags={'climatology_window':config['climatology_window'],'expected_snowfall_cm':f'{champion_expected_cm:.6f}','training_rows':len(training),'promotion_gate_passed':'true'},
        artifact_paths=artifacts)

    challenger_version, challenger_created = register_sklearn_version(
        client=client, registered_model_name=model_name, experiment_name=config['experiment_name'],
        role='challenger', method='ElasticNet + NAO', estimator=challenger_model, X_example=X_example,
        metrics=comparison['ElasticNet + NAO'], snapshot_hash=challenger_snapshot_hash,
        tags={'training_rows':len(training),'promotion_gate_passed':str(bool(decision['promote_challenger'])).lower(),'mae_skill_pct':decision['challenger_mae_skill_pct'],'rmse_skill_pct':decision['challenger_rmse_skill_pct']},
        artifact_paths=artifacts)

    champion_alias = config['champion_alias']
    challenger_alias = config['challenger_alias']
    client.set_registered_model_alias(model_name, challenger_alias, str(challenger_version.version))
    if decision['promote_challenger']:
        final_champion_version = challenger_version
        final_champion_method = 'ElasticNet + NAO'
    else:
        final_champion_version = champion_version
        final_champion_method = 'Historical Climatology'
    client.set_registered_model_alias(model_name, champion_alias, str(final_champion_version.version))

    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'tracking_uri': tracking_uri,
        'experiment_name': config['experiment_name'],
        'registered_model_name': model_name,
        'champion': {'alias':champion_alias,'version':str(final_champion_version.version),'method':final_champion_method},
        'challenger': {'alias':challenger_alias,'version':str(challenger_version.version),'method':'ElasticNet + NAO'},
        'promotion_gate_passed': bool(decision['promote_challenger']),
        'champion_version_created_this_run': champion_created,
        'challenger_version_created_this_run': challenger_created,
    }
    out = PROJECT_ROOT/'models'/'retraining'/'latest_mlflow_registry.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print()
    print('MLFLOW')
    print('------')
    print(f'Tracking DB: {tracking_db}')
    print(f'Registered model: {model_name}')
    print(f"Champion alias: {champion_alias} -> version {final_champion_version.version} ({final_champion_method})")
    print(f"Challenger alias: {challenger_alias} -> version {challenger_version.version} (ElasticNet + NAO)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
