import argparse
import json
import os
from typing import Literal, cast

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.config import TrainConfig
from src.features import add_engineered_features, make_X_y
from src.metrics import (
    scalar_metrics,
    precision_recall_at_k,
    threshold_f1,
    threshold_topk,      # informational cutoff in topk mode
    report_at_threshold,
    report_at_topk,
)

CalibrationMethod = Literal["sigmoid", "isotonic"]
ModelType = Literal["lr", "hgb"]
Policy = Literal["topk", "f1"]


def build_preprocess(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ],
        remainder="drop",
    )


def build_model(model_type: str, seed: int):
    mt = model_type.lower().strip()
    if mt == "lr":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if mt == "hgb":
        return HistGradientBoostingClassifier(
            random_state=seed,
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
        )
    raise ValueError(f"Unknown model_type={model_type}. Use 'lr' or 'hgb'.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PdM model on AI4I2020 dataset.")
    p.add_argument("--data-path", default=None, help="Path to CSV (default from config).")
    p.add_argument("--artifacts-dir", default=None, help="Output directory for artifacts.")
    p.add_argument("--model-type", default=None, choices=["lr", "hgb"], help="Model: lr or hgb.")
    p.add_argument("--no-calibrate", action="store_true", help="Disable probability calibration.")
    p.add_argument("--policy", default=None, choices=["topk", "f1"], help="Decision policy for reporting.")
    p.add_argument("--topk", type=int, default=None, help="k for top-k policy (default from config).")
    p.add_argument("--seed", type=int, default=None, help="Override random_state (split + model).")
    return p.parse_args()


def _cm_to_counts(cm: list[list[int]]) -> tuple[int, int, int, int]:
    tn, fp = cm[0]
    fn, tp = cm[1]
    return int(tn), int(fp), int(fn), int(tp)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    seed = args.seed if args.seed is not None else cfg.random_state

    # Override config from CLI
    data_path = args.data_path or cfg.data_path
    artifacts_dir = args.artifacts_dir or cfg.artifacts_dir
    model_type: ModelType = cast(ModelType, (args.model_type or cfg.model_type))
    calibrate = (not args.no_calibrate) and cfg.calibrate
    policy: Policy = cast(Policy, (args.policy or cfg.decision_policy))
    topk_k = args.topk or cfg.topk_k

    os.makedirs(artifacts_dir, exist_ok=True)

    # ---- Load + feature engineering ----
    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    # ---- Non-None columns (cfg.__post_init__ enforces this at runtime) ----
    cat_cols = cfg.cat_cols
    num_cols = cfg.num_cols
    drop_cols = cfg.drop_cols
    assert cat_cols is not None
    assert num_cols is not None
    assert drop_cols is not None

    # ---- Build X/y (leakage-safe) ----
    X, y = make_X_y(df, cfg.target, drop_cols, cat_cols, num_cols)

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=seed,
        stratify=y,
    )

    # ---- Pipeline ----
    preprocess = build_preprocess(cat_cols, num_cols)
    base_model = build_model(model_type, seed)

    base_pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", base_model),
    ])

    # ---- Fit (and calibrate optionally) ----
    if calibrate:
        method: CalibrationMethod = cast(CalibrationMethod, cfg.calibration_method)
        model = CalibratedClassifierCV(
            estimator=base_pipe,
            method=method,
            cv=cfg.calibration_cv,
        )
        model.fit(X_train, y_train)
    else:
        model = base_pipe
        model.fit(X_train, y_train)

    # ---- Evaluate ----
    proba = model.predict_proba(X_test)[:, 1]
    scalars = scalar_metrics(y_test, proba)

    # ---- Decision/reporting ----
    if policy == "f1":
        thr = threshold_f1(y_test, proba)
        decision_report = report_at_threshold(y_test, proba, thr)
        cutoff_or_threshold = float(decision_report["threshold"])
    else:
        decision_report = report_at_topk(y_test, proba, topk_k)
        # informational cutoff (ties may cause >=cutoff to select >k, but selection is exact-topk)
        cutoff_or_threshold = float(threshold_topk(proba, topk_k))

    # ---- Top-k summary at chosen k ----
    p_at_k, r_at_k = precision_recall_at_k(y_test.reset_index(drop=True), proba, k=topk_k)

    # ---- Capacity sweep (choose k) ----
    k_grid = [10, 20, 30, 40, 50, 75, 100]
    k_sweep = []
    for k in k_grid:
        if k > len(proba):
            continue
        rep_k = report_at_topk(y_test, proba, k)
        tn, fp, fn, tp = _cm_to_counts(rep_k["confusion_matrix"])

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0

        k_sweep.append({
            "k": int(k),
            "precision": float(prec),
            "recall": float(rec),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "cutoff_score": float(rep_k["cutoff_score"]),
            "num_ge_cutoff": int(rep_k["num_ge_cutoff"]),
        })

    results = {
        "data_path": data_path,
        "model_type": model_type,
        "calibrated": calibrate,
        "policy": policy,
        "topk_k": int(topk_k),
        "seed": int(seed),
        "metrics": scalars,
        "decision_report": decision_report,
        "cutoff_or_threshold": float(cutoff_or_threshold),
        "precision_at_k": float(p_at_k),
        "recall_at_k": float(r_at_k),
        "k_sweep": k_sweep,
        "features": {
            "categorical": cat_cols,
            "numerical": num_cols,
            "engineered": ["TempDiff", "Torque_x_RPM"],
            "dropped": drop_cols,
            "target": cfg.target,
        },
    }

    # ---- Save artifacts ----
    model_path = os.path.join(artifacts_dir, "pdm_model.joblib")
    meta_path = os.path.join(artifacts_dir, "pdm_meta.json")

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- Console summary ----
    print("Saved model:", model_path)
    print("Saved metadata:", meta_path)
    print("Seed:", seed)
    print("PR-AUC:", results["metrics"]["pr_auc"])
    print("ROC-AUC:", results["metrics"]["roc_auc"])
    print(f"Policy={policy}, cutoff/threshold={results['cutoff_or_threshold']}")
    print(f"precision@{topk_k}={results['precision_at_k']:.3f}, recall@{topk_k}={results['recall_at_k']:.3f}")

    if policy == "topk" and isinstance(decision_report, dict) and "num_ge_cutoff" in decision_report:
        print(
            f"Selected={decision_report['selected']} (should equal k), "
            f"num_scores>=cutoff={decision_report['num_ge_cutoff']} (tie diagnostic)"
        )

    print("\nK sweep (capacity trade-off):")
    for row in k_sweep:
        print(
            f"k={row['k']:>3} | precision={row['precision']:.3f} | recall={row['recall']:.3f} "
            f"| TP={row['tp']:>3} FP={row['fp']:>3} FN={row['fn']:>3} | ties>=cutoff={row['num_ge_cutoff']}"
        )


if __name__ == "__main__":
    main()
