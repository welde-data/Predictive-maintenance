import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physically meaningful engineered features used in PdM.
    Assumes raw AI4I columns exist in df.
    """
    out = df.copy()
    out["TempDiff"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Torque_x_RPM"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"]
    return out


def make_X_y(df: pd.DataFrame, target: str, drop_cols: list, cat_cols: list, num_cols: list):
    """
    Leakage-safe feature matrix and target vector.
    """
    y = df[target].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X[cat_cols + num_cols]
    return X, y
