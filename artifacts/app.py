import os
import io
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import TrainConfig
from features import add_engineered_features

st.set_page_config(page_title="Predictive Maintenance — Scoring", layout="wide")


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def safe_pick_id_cols(df: pd.DataFrame) -> list[str]:
    candidates = ["UDI", "Product ID"]
    return [c for c in candidates if c in df.columns]


def build_X(df_fe: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    cat_cols = cfg.cat_cols
    num_cols = cfg.num_cols
    drop_cols = cfg.drop_cols
    assert cat_cols is not None and num_cols is not None and drop_cols is not None

    required = list(cat_cols + num_cols)
    missing = [c for c in required if c not in df_fe.columns]
    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")

    X = df_fe.drop(columns=[c for c in drop_cols if c in df_fe.columns], errors="ignore")
    X = X[required]
    return X


def score_df(df_raw: pd.DataFrame, model, cfg: TrainConfig, k: int) -> pd.DataFrame:
    df_fe = add_engineered_features(df_raw)
    X = build_X(df_fe, cfg)

    proba = model.predict_proba(X)[:, 1]

    scored = df_raw.copy()
    scored["p_failure"] = proba
    scored["rank"] = scored["p_failure"].rank(method="first", ascending=False).astype(int)
    scored["alert_topk"] = scored["rank"] <= int(k)

    # Engineered features for interpretability
    scored["TempDiff"] = df_fe["TempDiff"]
    scored["Torque_x_RPM"] = df_fe["Torque_x_RPM"]

    # Reorder columns
    id_cols = safe_pick_id_cols(scored)
    front = id_cols + ["p_failure", "rank", "alert_topk", "TempDiff", "Torque_x_RPM"]
    front = [c for c in front if c in scored.columns]
    rest = [c for c in scored.columns if c not in front]
    scored = scored[front + rest]

    scored = scored.sort_values("rank", ascending=True).reset_index(drop=True)
    return scored


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def filter_scored(scored: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """
    Apply UI filters to the scored dataframe. Filters are optional and only apply if columns exist.
    """
    out = scored.copy()

    # Minimum probability
    min_p = st.session_state.get("min_p_failure", 0.0)
    out = out[out["p_failure"] >= float(min_p)]

    # Type filter (if present)
    if cfg.cat_cols and "Type" in out.columns:
        selected_types = st.session_state.get("type_filter", None)
        if selected_types:
            out = out[out["Type"].isin(selected_types)]

    # Numeric range filters (only if column exists)
    range_specs = [
        ("Tool wear [min]", "tool_wear_range"),
        ("Torque [Nm]", "torque_range"),
        ("Rotational speed [rpm]", "rpm_range"),
        ("TempDiff", "tempdiff_range"),
    ]
    for col, key in range_specs:
        if col in out.columns and key in st.session_state:
            lo, hi = st.session_state[key]
            out = out[(out[col] >= lo) & (out[col] <= hi)]

    # Alerts only
    alerts_only = st.session_state.get("alerts_only", True)
    if alerts_only:
        out = out[out["alert_topk"]]

    return out


def main() -> None:
    cfg = TrainConfig()

    st.title("Predictive Maintenance — Scoring & Top-K Alerts")

    with st.sidebar:
        st.header("Model & scoring")

        artifacts_dir = st.text_input("Artifacts directory", value=cfg.artifacts_dir)
        model_path = st.text_input(
            "Model path",
            value=os.path.join(artifacts_dir, "pdm_model.joblib"),
        )
        topk = st.number_input("Top-K alerts", min_value=1, max_value=100000, value=int(cfg.topk_k), step=1)

        st.divider()
        st.header("Input data")

        mode = st.radio("Input mode", options=["Upload CSV", "Local file path"], index=0)

        upload = None
        local_path = None

        if mode == "Upload CSV":
            upload = st.file_uploader("Upload CSV to score", type=["csv"])
        else:
            local_path = st.text_input("CSV path on this machine", value=cfg.data_path)

        st.caption("The CSV must contain the columns used in training (Type + numeric sensors).")

        st.divider()
        st.header("Filters")

        st.checkbox("Show only alerts", value=True, key="alerts_only")

        st.slider(
            "Minimum p_failure",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="min_p_failure",
        )

    # Load model
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.stop()

    model = load_model(model_path)

    # Load data
    df_raw = None
    if mode == "Upload CSV":
        if upload is None:
            st.info("Upload a CSV to score.")
            st.stop()
        try:
            df_raw = pd.read_csv(upload)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()
    else:
        if not local_path:
            st.info("Provide a local CSV path to score.")
            st.stop()
        if not os.path.exists(local_path):
            st.error(f"CSV not found at: {local_path}")
            st.stop()
        try:
            df_raw = pd.read_csv(local_path)
        except Exception as e:
            st.error(f"Could not read CSV at path: {e}")
            st.stop()

    # Score
    try:
        scored = score_df(df_raw, model, cfg, int(topk))
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Build dynamic filter widgets (after scoring, so engineered columns exist)
    with st.sidebar:
        # Type multi-select
        if "Type" in scored.columns:
            types = sorted(scored["Type"].dropna().astype(str).unique().tolist())
            st.multiselect("Type", options=types, default=types, key="type_filter")

        # Numeric range filters (only show if column exists)
        def range_slider(col: str, key: str):
            if col not in scored.columns:
                return
            s = scored[col].dropna()
            if s.empty:
                return
            lo = float(s.min())
            hi = float(s.max())
            # Default full range
            st.slider(col, min_value=lo, max_value=hi, value=(lo, hi), key=key)

        range_slider("Tool wear [min]", "tool_wear_range")
        range_slider("Torque [Nm]", "torque_range")
        range_slider("Rotational speed [rpm]", "rpm_range")
        range_slider("TempDiff", "tempdiff_range")

    # Summary metrics
    id_cols = safe_pick_id_cols(df_raw)
    n = len(scored)
    n_alerts = int(scored["alert_topk"].sum())
    cutoff = float(scored.loc[scored["alert_topk"], "p_failure"].min()) if n_alerts > 0 else float("nan")
    n_ones = int((scored["p_failure"] == 1.0).sum())
    n_unique = int(scored["p_failure"].nunique())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{n:,}")
    m2.metric("Top-K", f"{int(topk):,}")
    m3.metric("Alerts selected", f"{n_alerts:,}")
    m4.metric("Cutoff score (Top-K)", f"{cutoff:.6f}" if n_alerts > 0 else "—")
    m5.metric("Unique p values", f"{n_unique:,}")

    st.caption(f"ID columns found: {', '.join(id_cols) if id_cols else 'None'} | p_failure==1.0 count: {n_ones:,}")

    # Filtered view
    filtered = filter_scored(scored, cfg)

    st.divider()

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Results table")
        st.dataframe(filtered, use_container_width=True, height=560)

        st.download_button(
            label="Download scored CSV (full, unfiltered)",
            data=to_csv_bytes(scored),
            file_name="pdm_scored.csv",
            mime="text/csv",
        )

        st.download_button(
            label="Download filtered CSV (current view)",
            data=to_csv_bytes(filtered),
            file_name="pdm_scored_filtered.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("Score distribution")

        fig = plt.figure()
        plt.hist(scored["p_failure"].to_numpy(dtype=float), bins=30)
        plt.xlabel("p_failure")
        plt.ylabel("count")
        st.pyplot(fig, clear_figure=True)

        st.subheader("Top alerts preview")
        preview_cols = (id_cols + ["p_failure", "rank"]) if id_cols else ["p_failure", "rank"]
        st.dataframe(
            scored.loc[scored["alert_topk"], preview_cols].head(10),
            use_container_width=True,
            height=300,
        )

        with st.expander("Scoring schema"):
            st.write({
                "categorical": cfg.cat_cols,
                "numerical": cfg.num_cols,
                "dropped": cfg.drop_cols,
                "target": cfg.target,
            })


if __name__ == "__main__":
    main()
