import argparse
import os
from typing import Optional

import joblib
import pandas as pd

from src.config import TrainConfig
from src.features import add_engineered_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score PdM model and produce top-k alerts.")
    p.add_argument("--data-path", required=True, help="Path to input CSV to score.")
    p.add_argument("--model-path", default=None, help="Path to joblib model (default artifacts/pdm_model.joblib).")
    p.add_argument("--artifacts-dir", default=None, help="Output directory (default from config).")
    p.add_argument("--topk", type=int, default=None, help="Top-k alerts to flag (default from config).")
    p.add_argument("--out-csv", default=None, help="Output CSV filename (default: pdm_scored.csv).")
    return p.parse_args()


def _safe_pick_id_cols(df: pd.DataFrame) -> list[str]:
    """
    Pick identifier columns if present (for review/ops).
    """
    candidates = ["UDI", "Product ID"]
    return [c for c in candidates if c in df.columns]


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    data_path = args.data_path
    artifacts_dir = args.artifacts_dir or cfg.artifacts_dir
    model_path = args.model_path or os.path.join(artifacts_dir, "pdm_model.joblib")
    topk_k = args.topk or cfg.topk_k
    out_name = args.out_csv or "pdm_scored.csv"
    out_path = os.path.join(artifacts_dir, out_name)

    os.makedirs(artifacts_dir, exist_ok=True)

    # Load model
    model = joblib.load(model_path)

    # Load data
    df = pd.read_csv(data_path)
    df_fe = add_engineered_features(df)

    # Ensure config columns are present (cfg.__post_init__ sets them)
    cat_cols = cfg.cat_cols
    num_cols = cfg.num_cols
    drop_cols = cfg.drop_cols
    assert cat_cols is not None
    assert num_cols is not None
    assert drop_cols is not None

    # Build X (same as training, but no y)
    X = df_fe.drop(columns=[c for c in drop_cols if c in df_fe.columns], errors="ignore")
    X = X[cat_cols + num_cols]

    # Score
    proba = model.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["p_failure"] = proba
    scored["rank"] = scored["p_failure"].rank(method="first", ascending=False).astype(int)
    scored["alert_topk"] = scored["rank"] <= int(topk_k)

    # Put key columns first for readability
    id_cols = _safe_pick_id_cols(scored)
    front = id_cols + ["p_failure", "rank", "alert_topk"]
    other_cols = [c for c in scored.columns if c not in front]
    scored = scored[front + other_cols]

    # Sort by rank
    scored = scored.sort_values("rank", ascending=True)

    scored.to_csv(out_path, index=False)

    # Console summary
    n = len(scored)
    n_alert = int(scored["alert_topk"].sum())
    print("Loaded model:", model_path)
    print("Scored rows:", n)
    print(f"Top-k alerts: {n_alert} (k={topk_k})")
    print("Saved scored output:", out_path)

    # Show the top 5 alerts
    show_cols = (id_cols + ["p_failure", "rank"]) if id_cols else ["p_failure", "rank"]
    print("\nTop 5 alerts preview:")
    print(scored.loc[scored["alert_topk"], show_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
