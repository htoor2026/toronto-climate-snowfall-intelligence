from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# App configuration
# ============================================================

st.set_page_config(
    page_title="Toronto Snowfall Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent

FORECAST_PATH = PROJECT_ROOT / "forecasts" / "latest_forecast.json"
FORECAST_HISTORY_PATH = PROJECT_ROOT / "forecasts" / "forecast_history.jsonl"

RETRAINING_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retraining_model_comparison.csv"
)

RECENT_METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retraining_recent_comparison.csv"
)

RETRAINING_DECISION_PATH = (
    PROJECT_ROOT
    / "models"
    / "retraining"
    / "latest_retraining_decision.json"
)

MLFLOW_REGISTRY_PATH = (
    PROJECT_ROOT
    / "models"
    / "retraining"
    / "latest_mlflow_registry.json"
)

AGENT_EVENT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "agent"
    / "latest_agent_event.json"
)

AGENT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "agent"
    / "latest_agent_report.md"
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .hero {
            padding: 1.25rem 1.4rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.78;
            margin-bottom: 0;
        }

        .info-box {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 12px;
            margin: 0.5rem 0 1rem 0;
        }

        .small-muted {
            opacity: 0.72;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            padding: 0.85rem;
            border-radius: 12px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data helpers
# ============================================================

def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records


def fmt_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def file_mtime(path: Path) -> str:
    if not path.exists():
        return "missing"

    return datetime.fromtimestamp(
        path.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M")


def season_label(season_year: Any) -> str:
    try:
        year = int(season_year)
        return f"{year - 1}-{str(year)[-2:]}"
    except (TypeError, ValueError):
        return "unknown"


def extract_change_block(snapshot: dict[str, Any]) -> dict[str, Any]:
    change = snapshot.get("change_detection", {}) or {}

    if "production" in change or "challenger" in change:
        return change

    return {
        "material": bool(change.get("material")),
        "production": {
            "material": bool(change.get("material")),
            "reasons": change.get("reasons", []),
            "metrics": change.get("metrics", {}),
        },
        "challenger": {
            "material": False,
            "reasons": [],
            "metrics": {},
        },
    }


def build_history_dataframe(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for index, snapshot in enumerate(records, start=1):
        production = snapshot.get("production_forecast", {}) or {}
        challenger = snapshot.get("challenger_forecast", {}) or {}
        change = extract_change_block(snapshot)

        generated = (
            snapshot.get("generated_at_utc")
            or snapshot.get("timestamp_utc")
            or snapshot.get("created_at_utc")
        )

        rows.append(
            {
                "snapshot": index,
                "generated_at": generated,
                "season": season_label(
                    snapshot.get("season_year")
                ),
                "production_expected_cm": production.get(
                    "expected_snowfall_cm"
                ),
                "production_low_cm": production.get(
                    "calibrated_lower80_cm"
                ),
                "production_high_cm": production.get(
                    "calibrated_upper80_cm"
                ),
                "challenger_expected_cm": challenger.get(
                    "expected_snowfall_cm"
                ),
                "challenger_status": challenger.get(
                    "status",
                    "unknown",
                ),
                "material_change": bool(
                    change.get("material")
                ),
            }
        )

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    if "generated_at" in frame.columns:
        parsed = pd.to_datetime(
            frame["generated_at"],
            errors="coerce",
            utc=True,
        )

        if parsed.notna().any():
            frame["generated_at"] = parsed

    return frame


forecast = load_json(FORECAST_PATH)
history_records = load_jsonl(FORECAST_HISTORY_PATH)
history_df = build_history_dataframe(history_records)

retraining_metrics = load_csv(RETRAINING_METRICS_PATH)
recent_metrics = load_csv(RECENT_METRICS_PATH)
retraining_decision = load_json(RETRAINING_DECISION_PATH)
mlflow_registry = load_json(MLFLOW_REGISTRY_PATH)
agent_event = load_json(AGENT_EVENT_PATH)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("System Snapshot")

    if forecast:
        production = forecast.get(
            "production_forecast",
            {},
        ) or {}

        challenger = forecast.get(
            "challenger_forecast",
            {},
        ) or {}

        st.caption("Current season")
        st.write(
            f"**{season_label(forecast.get('season_year'))}**"
        )

        st.caption("Production")
        st.write(
            production.get(
                "method",
                "Unknown",
            )
        )

        st.caption("Challenger")
        st.write(
            challenger.get(
                "status",
                "unknown",
            )
        )

        change = extract_change_block(forecast)

        st.caption("Material change")
        st.write(
            "Yes"
            if change.get("material")
            else "No"
        )
    else:
        st.warning(
            "latest_forecast.json was not found."
        )

    st.divider()

    st.caption("Latest file updates")
    st.write(
        f"Forecast: {file_mtime(FORECAST_PATH)}"
    )
    st.write(
        f"Retraining: {file_mtime(RETRAINING_DECISION_PATH)}"
    )
    st.write(
        f"Registry: {file_mtime(MLFLOW_REGISTRY_PATH)}"
    )

    if st.button(
        "Reload dashboard",
        use_container_width=True,
    ):
        st.rerun()


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Toronto Snowfall Intelligence
        </div>
        <div class="hero-subtitle">
            Historical climate analysis, seasonal snowfall forecasting,
            uncertainty, retraining, model governance and operational monitoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tabs = st.tabs(
    [
        "Current Forecast",
        "Forecast History",
        "Model & MLOps",
        "Business Communication",
        "Technical Walkthrough",
        "System Status",
    ]
)


# ============================================================
# Tab 1 - Current Forecast
# ============================================================

with tabs[0]:
    st.header("Current Seasonal Snowfall Forecast")

    if not forecast:
        st.error(
            "No forecast snapshot is available."
        )
    else:
        production = forecast.get(
            "production_forecast",
            {},
        ) or {}

        challenger = forecast.get(
            "challenger_forecast",
            {},
        ) or {}

        live = forecast.get(
            "live_predictors",
            {},
        ) or {}

        change = extract_change_block(
            forecast
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Expected snowfall",
                (
                    f"{fmt_number(production.get('expected_snowfall_cm'))} cm"
                ),
            )

        with c2:
            st.metric(
                "Median",
                (
                    f"{fmt_number(production.get('median_snowfall_cm'))} cm"
                ),
            )

        with c3:
            lower = fmt_number(
                production.get(
                    "calibrated_lower80_cm"
                )
            )
            upper = fmt_number(
                production.get(
                    "calibrated_upper80_cm"
                )
            )

            st.metric(
                "Calibrated 80% interval",
                f"{lower}–{upper} cm",
            )

        with c4:
            st.metric(
                "Material change",
                (
                    "Yes"
                    if change.get("material")
                    else "No"
                ),
            )

        st.subheader(
            "Below / Near / Above Normal Probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Probability": [
                    float(
                        production.get(
                            "p_below",
                            0,
                        )
                        or 0
                    ),
                    float(
                        production.get(
                            "p_near",
                            0,
                        )
                        or 0
                    ),
                    float(
                        production.get(
                            "p_above",
                            0,
                        )
                        or 0
                    ),
                ]
            },
            index=[
                "Below normal",
                "Near normal",
                "Above normal",
            ],
        )

        st.bar_chart(
            probability_df,
            horizontal=True,
            height=260,
        )

        threshold_cols = st.columns(3)

        threshold_cols[0].metric(
            "Below-normal threshold",
            (
                f"{fmt_number(production.get('below_normal_threshold_cm'))} cm"
            ),
        )

        threshold_cols[1].metric(
            "Above-normal threshold",
            (
                f"{fmt_number(production.get('above_normal_threshold_cm'))} cm"
            ),
        )

        threshold_cols[2].metric(
            "Calibration adjustment",
            (
                f"{fmt_number(production.get('conformal_qhat_cm'))} cm"
            ),
        )

        st.divider()

        left, right = st.columns(
            [1.15, 1]
        )

        with left:
            st.subheader(
                "Production vs Challenger"
            )

            comparison_rows = [
                {
                    "Role": "Production",
                    "Method": production.get(
                        "method"
                    ),
                    "Status": production.get(
                        "status"
                    ),
                    "Expected snowfall (cm)": production.get(
                        "expected_snowfall_cm"
                    ),
                },
                {
                    "Role": "Challenger",
                    "Method": challenger.get(
                        "method"
                    ),
                    "Status": challenger.get(
                        "status"
                    ),
                    "Expected snowfall (cm)": challenger.get(
                        "expected_snowfall_cm"
                    ),
                },
            ]

            st.dataframe(
                pd.DataFrame(
                    comparison_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

            if challenger.get(
                "status"
            ) == "not_ready":
                st.info(
                    "The challenger is intentionally blocked until "
                    "its required climate and Sep–Oct predictors are complete."
                )

        with right:
            st.subheader(
                "Live Predictor Readiness"
            )

            predictor_rows = [
                {
                    "Predictor": "ONI JAS",
                    "Value": live.get(
                        "oni_jas"
                    ),
                },
                {
                    "Predictor": "NAO JAS",
                    "Value": live.get(
                        "nao_jas"
                    ),
                },
                {
                    "Predictor": "Sep–Oct mean temperature",
                    "Value": live.get(
                        "sep_oct_mean_temp_c"
                    ),
                },
                {
                    "Predictor": "Sep–Oct precipitation",
                    "Value": live.get(
                        "sep_oct_total_precip_mm"
                    ),
                },
                {
                    "Predictor": "Temperature coverage %",
                    "Value": live.get(
                        "sep_oct_temp_coverage_pct"
                    ),
                },
                {
                    "Predictor": "Precipitation coverage %",
                    "Value": live.get(
                        "sep_oct_precip_coverage_pct"
                    ),
                },
            ]

            st.dataframe(
                pd.DataFrame(
                    predictor_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The production forecast currently uses probabilistic climatology. "
            "The climate-feature model remains a challenger because it has not "
            "demonstrated enough out-of-sample skill to replace the benchmark."
        )


# ============================================================
# Tab 2 - Forecast History
# ============================================================

with tabs[1]:
    st.header("Forecast History")

    if history_df.empty:
        st.info(
            "No forecast history is available yet."
        )
    else:
        st.write(
            "Each refresh is archived so forecast changes can be audited over time."
        )

        chart_df = (
            history_df[
                [
                    "snapshot",
                    "production_expected_cm",
                    "challenger_expected_cm",
                ]
            ]
            .set_index("snapshot")
        )

        st.line_chart(
            chart_df,
            height=360,
        )

        material_count = int(
            history_df[
                "material_change"
            ].sum()
        )

        h1, h2, h3 = st.columns(3)

        h1.metric(
            "Stored snapshots",
            len(history_df),
        )

        h2.metric(
            "Material-change snapshots",
            material_count,
        )

        h3.metric(
            "Latest challenger status",
            str(
                history_df[
                    "challenger_status"
                ].iloc[-1]
            ),
        )

        display_history = (
            history_df.copy()
            .sort_values(
                "snapshot",
                ascending=False,
            )
        )

        if pd.api.types.is_datetime64_any_dtype(
            display_history.get(
                "generated_at"
            )
        ):
            display_history[
                "generated_at"
            ] = display_history[
                "generated_at"
            ].dt.strftime(
                "%Y-%m-%d %H:%M UTC"
            )

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Tab 3 - Model & MLOps
# ============================================================

with tabs[2]:
    st.header("Model Governance and MLOps")

    if retraining_decision:
        promote = bool(
            retraining_decision.get(
                "promote_challenger"
            )
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Champion",
            retraining_decision.get(
                "champion",
                "unknown",
            ),
        )

        m2.metric(
            "Challenger",
            retraining_decision.get(
                "challenger",
                "unknown",
            ),
        )

        m3.metric(
            "MAE skill",
            (
                f"{fmt_number(retraining_decision.get('challenger_mae_skill_pct'), 2)}%"
            ),
        )

        m4.metric(
            "Promotion decision",
            (
                "Promote"
                if promote
                else "Retain champion"
            ),
        )

        st.markdown(
            f"""
            <div class="info-box">
                <strong>Latest decision:</strong>
                {retraining_decision.get('decision_reason', 'No reason recorded.')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if retraining_metrics is not None:
        st.subheader(
            "Walk-Forward Model Comparison"
        )

        st.dataframe(
            retraining_metrics,
            use_container_width=True,
            hide_index=True,
        )

        if {
            "model",
            "MAE_cm",
            "RMSE_cm",
        }.issubset(
            retraining_metrics.columns
        ):
            chart = retraining_metrics[
                [
                    "model",
                    "MAE_cm",
                    "RMSE_cm",
                ]
            ].set_index(
                "model"
            )

            st.bar_chart(
                chart,
                height=330,
            )

    if recent_metrics is not None:
        with st.expander(
            "Recent 15-winter comparison"
        ):
            st.dataframe(
                recent_metrics,
                use_container_width=True,
                hide_index=True,
            )

    if mlflow_registry:
        st.subheader(
            "MLflow Registry"
        )

        champion = (
            mlflow_registry.get(
                "champion",
                {},
            )
            or {}
        )

        challenger = (
            mlflow_registry.get(
                "challenger",
                {},
            )
            or {}
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Champion alias",
            (
                f"v{champion.get('version', '?')}"
            ),
            champion.get(
                "method",
                "unknown",
            ),
        )

        r2.metric(
            "Challenger alias",
            (
                f"v{challenger.get('version', '?')}"
            ),
            challenger.get(
                "method",
                "unknown",
            ),
        )

        r3.metric(
            "Promotion gate",
            (
                "Passed"
                if mlflow_registry.get(
                    "promotion_gate_passed"
                )
                else "Failed"
            ),
        )

        st.caption(
            "A challenger is registered even when it is not promoted. "
            "This preserves experiment history without allowing a weaker "
            "model to replace production."
        )


# ============================================================
# Tab 4 - Business Communication
# ============================================================

with tabs[3]:
    st.header(
        "Business Communication"
    )

    st.write(
        "This section explains the project without requiring knowledge of "
        "machine learning, statistical testing, MLflow or model calibration."
    )

    st.subheader(
        "1. What business problem does this solve?"
    )

    st.markdown(
        """
        Seasonal snowfall creates planning uncertainty for organizations that
        depend on weather-sensitive operations. Examples include municipal snow
        removal, airport operations, logistics, staffing, maintenance planning,
        retail inventory and transportation.

        The goal of this project is not to claim that a single model can predict
        Toronto snowfall perfectly. The goal is to turn historical climate data
        and current seasonal signals into a decision-support system that answers:

        - What is a reasonable snowfall expectation for the coming winter?
        - How uncertain is that estimate?
        - Is the season more likely to be below, near or above normal?
        - Have new climate signals changed the outlook enough to matter?
        - Should a newer forecasting model replace the current production method?
        """
    )

    st.subheader(
        "2. What does the system deliver?"
    )

    if forecast:
        production = (
            forecast.get(
                "production_forecast",
                {},
            )
            or {}
        )

        st.markdown(
            f"""
            For the **{season_label(forecast.get('season_year'))}** season,
            the current production system estimates approximately
            **{fmt_number(production.get('expected_snowfall_cm'))} cm**
            of snowfall.

            The calibrated 80% uncertainty range is approximately
            **{fmt_number(production.get('calibrated_lower80_cm'))} to
            {fmt_number(production.get('calibrated_upper80_cm'))} cm**.

            Instead of presenting one number as certainty, the system also
            communicates the probability of a below-normal, near-normal or
            above-normal winter.
            """
        )

    st.subheader(
        "3. What did the historical analysis find?"
    )

    findings = pd.DataFrame(
        [
            {
                "Finding": "Long-term temperature",
                "Business interpretation": (
                    "Toronto has shown a clear long-term warming signal, "
                    "so historical operating assumptions should not be treated "
                    "as permanently fixed."
                ),
            },
            {
                "Finding": "Hot and cold extremes",
                "Business interpretation": (
                    "Temperature extremes are changing, which matters for "
                    "planning, resilience and seasonal operations."
                ),
            },
            {
                "Finding": "Total precipitation",
                "Business interpretation": (
                    "The project did not find strong evidence of a simple "
                    "linear trend in total annual precipitation."
                ),
            },
            {
                "Finding": "Seasonal snowfall",
                "Business interpretation": (
                    "Historical snowfall is highly variable, and the analysis "
                    "did not support a strong simple long-term linear snowfall trend."
                ),
            },
        ]
    )

    st.dataframe(
        findings,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "4. Why is the simple model still in production?"
    )

    st.markdown(
        """
        The machine-learning challenger uses climate signals such as ENSO, NAO
        and autumn weather conditions. It is more sophisticated than the
        climatology benchmark, but sophistication alone is not a reason to deploy it.

        In walk-forward testing, the challenger improved average absolute error
        only slightly and produced a slightly worse RMSE than climatology. The
        project therefore keeps the simpler climatology model as the champion.

        This is an important business decision: the system prefers demonstrated
        performance and stability over unnecessary model complexity.
        """
    )

    st.subheader(
        "5. How would a business use this?"
    )

    use_cases = pd.DataFrame(
        [
            {
                "Decision": "Seasonal resource planning",
                "Use": (
                    "Use expected snowfall and uncertainty to estimate staffing, "
                    "equipment and contractor needs."
                ),
            },
            {
                "Decision": "Risk planning",
                "Use": (
                    "Use the wide forecast interval to plan for downside and "
                    "high-snow scenarios rather than relying on one point estimate."
                ),
            },
            {
                "Decision": "Operational monitoring",
                "Use": (
                    "Refresh climate predictors and alert only when the outlook "
                    "changes materially."
                ),
            },
            {
                "Decision": "Model governance",
                "Use": (
                    "Promote a challenger only after it passes predefined "
                    "out-of-sample performance rules."
                ),
            },
        ]
    )

    st.dataframe(
        use_cases,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "6. Executive summary"
    )

    st.markdown(
        """
        **The business value is not just the snowfall prediction.**
        The value is the full decision process:

        historical evidence → forecast → uncertainty → monitoring →
        retraining → objective model comparison → governed promotion → alerting.

        The project demonstrates how a data product can communicate uncertainty,
        resist unnecessary complexity and support repeatable operational decisions.
        """
    )


# ============================================================
# Tab 5 - Technical Walkthrough
# ============================================================

with tabs[4]:
    st.header(
        "Technical Walkthrough"
    )

    st.write(
        "This section is designed for data scientists, ML engineers and technical reviewers."
    )

    st.subheader(
        "1. Data layer"
    )

    st.markdown(
        """
        **Historical climate analysis**
        - Environment and Climate Change Canada / homogenized climate data
        - Toronto City Centre for long-term climate analysis

        **Seasonal snowfall target**
        - Toronto Pearson station thread
        - Legacy station and current station are stitched across the 2013 transition
        - Seasonal target uses Oct–Apr snowfall
        - Strict coverage rules are applied before a winter can become a training target

        **Climate predictors**
        - NOAA Oceanic Niño Index (ONI)
        - North Atlantic Oscillation (NAO)
        - Sep–Oct Toronto temperature and precipitation
        - Lagged and rolling snowfall features
        """
    )

    st.subheader(
        "2. Leakage-safe feature engineering"
    )

    st.code(
        """
Forecast issue date: approximately November 1

Season ending 2027
    predictor year = 2026

Allowed:
    JAS 2026 ONI
    JAS 2026 NAO
    Sep-Oct 2026 Toronto weather
    snowfall history through 2025-26

Not allowed:
    any information from winter 2026-27 itself
        """.strip(),
        language="text",
    )

    st.markdown(
        """
        Lag and rolling snowfall features are built on the full calendar sequence
        before model-ready filtering. This prevents an unusable historical season
        from accidentally causing a multi-year jump in the lag definition.
        """
    )

    st.subheader(
        "3. Validation design"
    )

    st.markdown(
        """
        The project uses **one-step expanding walk-forward validation** rather than
        a random train/test split.

        For each forecast winter:

        1. Train only on winters that occurred earlier.
        2. Generate a forecast for the next winter.
        3. Record the error.
        4. Expand the training window by one season.
        5. Repeat through the available historical record.

        This reproduces the information constraint of a real forecasting system.
        """
    )

    st.subheader(
        "4. Models"
    )

    model_table = pd.DataFrame(
        [
            {
                "Role": "Champion",
                "Model": "Historical Climatology",
                "Purpose": (
                    "Strong, transparent benchmark and current production model."
                ),
            },
            {
                "Role": "Challenger",
                "Model": "ElasticNet + ONI + NAO + autumn climate + snowfall history",
                "Purpose": (
                    "Tests whether climate signals add stable out-of-sample skill."
                ),
            },
            {
                "Role": "Probability layer",
                "Model": "Empirical probabilistic climatology",
                "Purpose": (
                    "Produces below / near / above-normal probabilities."
                ),
            },
            {
                "Role": "Uncertainty layer",
                "Model": "Conformalized empirical interval",
                "Purpose": (
                    "Adjusts interval width using historical coverage performance."
                ),
            },
        ]
    )

    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "5. Promotion rule"
    )

    st.code(
        """
Promote challenger only if:

    MAE improvement vs climatology >= 3%
    AND
    RMSE does not degrade
        """.strip(),
        language="text",
    )

    st.markdown(
        """
        The current ElasticNet challenger improves MAE by less than the required
        threshold and slightly worsens RMSE, so it remains a challenger.
        """
    )

    st.subheader(
        "6. Forecast refresh vs retraining"
    )

    architecture = pd.DataFrame(
        [
            {
                "Workflow": "Forecast refresh",
                "Trigger": "New ONI / NAO / Sep–Oct information",
                "Action": (
                    "Refresh current-season predictors and forecast. "
                    "No model training."
                ),
            },
            {
                "Workflow": "Retraining",
                "Trigger": "A winter finishes and actual snowfall becomes known",
                "Action": (
                    "Append a new labelled season, rerun walk-forward validation, "
                    "fit challenger candidate and apply promotion gate."
                ),
            },
        ]
    )

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "7. MLOps and governance"
    )

    st.code(
        """
new climate data
    ↓
refresh_forecast.py
    ↓
production + challenger forecast
    ↓
change detection
    ↓
forecast history

completed winter
    ↓
retrain_model.py
    ↓
walk-forward evaluation
    ↓
promotion gate
    ↓
MLflow
    ├── @champion
    └── @challenger
    ↓
agent report / alert
        """.strip(),
        language="text",
    )

    st.markdown(
        """
        MLflow stores experiment metrics and registered model versions.
        The registry keeps separate `champion` and `challenger` aliases.
        A challenger can be registered without being promoted.
        """
    )

    st.subheader(
        "8. Agent behavior"
    )

    st.markdown(
        """
        The agent is intentionally deterministic and small. It reads existing
        system outputs and decides whether a material operational event occurred.

        It can generate a report or optional email alert, but it cannot:

        - change the target
        - change validation rules
        - alter promotion thresholds
        - retrain on its own
        - promote a challenger that failed the gate
        """
    )

    st.subheader(
        "9. Testing"
    )

    st.markdown(
        """
        Automated tests cover forecast change detection, challenger transitions,
        retraining decisions, MLflow registry helpers and agent policy.
        The current project test suite is designed to be run with:
        """
    )

    st.code(
        "python -m pytest tests/ -v",
        language="bash",
    )


# ============================================================
# Tab 6 - System Status
# ============================================================

with tabs[5]:
    st.header(
        "System Status"
    )

    status_rows = [
        {
            "Component": "Current forecast",
            "File": "forecasts/latest_forecast.json",
            "Status": (
                "Ready"
                if FORECAST_PATH.exists()
                else "Missing"
            ),
            "Last modified": file_mtime(
                FORECAST_PATH
            ),
        },
        {
            "Component": "Forecast history",
            "File": "forecasts/forecast_history.jsonl",
            "Status": (
                "Ready"
                if FORECAST_HISTORY_PATH.exists()
                else "Missing"
            ),
            "Last modified": file_mtime(
                FORECAST_HISTORY_PATH
            ),
        },
        {
            "Component": "Retraining decision",
            "File": "models/retraining/latest_retraining_decision.json",
            "Status": (
                "Ready"
                if RETRAINING_DECISION_PATH.exists()
                else "Missing"
            ),
            "Last modified": file_mtime(
                RETRAINING_DECISION_PATH
            ),
        },
        {
            "Component": "MLflow registry snapshot",
            "File": "models/retraining/latest_mlflow_registry.json",
            "Status": (
                "Ready"
                if MLFLOW_REGISTRY_PATH.exists()
                else "Missing"
            ),
            "Last modified": file_mtime(
                MLFLOW_REGISTRY_PATH
            ),
        },
        {
            "Component": "Agent event",
            "File": "reports/agent/latest_agent_event.json",
            "Status": (
                "Ready"
                if AGENT_EVENT_PATH.exists()
                else "Missing"
            ),
            "Last modified": file_mtime(
                AGENT_EVENT_PATH
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(
            status_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Latest Agent Event"
    )

    if agent_event:
        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Should report",
            str(
                bool(
                    agent_event.get(
                        "should_report"
                    )
                )
            ),
        )

        a2.metric(
            "Should email",
            str(
                bool(
                    agent_event.get(
                        "should_email"
                    )
                )
            ),
        )

        a3.metric(
            "Event count",
            len(
                agent_event.get(
                    "event_types",
                    [],
                )
            ),
        )

        st.json(
            agent_event
        )
    else:
        st.info(
            "No agent event file is available."
        )

    st.subheader(
        "Operational Commands"
    )

    st.code(
        """
# Refresh the current forecast
python scripts/refresh_forecast.py

# Check whether a new completed winter requires retraining
python scripts/retrain_model.py

# Manually rerun the benchmark
python scripts/retrain_model.py --force

# Update MLflow registry
python scripts/register_mlflow_models.py

# Run the operational agent
python scripts/run_agent.py

# Run automated tests
python -m pytest tests/ -v
        """.strip(),
        language="bash",
    )

    st.caption(
        "The dashboard is read-only by design. Operational workflows remain "
        "explicit CLI processes, which keeps this portfolio project auditable "
        "and avoids unnecessary application complexity."
    )
