from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainConfig:
    # IO
    data_path: str = "data/raw/ai4i.csv"
    artifacts_dir: str = "artifacts"

    # Target + columns
    target: str = "Machine failure"
    cat_cols: list[str] | None = None
    num_cols: list[str] | None = None

    # Columns to exclude (IDs + leakage)
    drop_cols: list[str] | None = None

    # Train/test split
    test_size: float = 0.2
    random_state: int = 42

    # Model selection
    model_type: Literal["lr", "hgb"] = "hgb"

    # Calibration
    calibrate: bool = True
    calibration_method: Literal["sigmoid", "isotonic"] = "isotonic"
    calibration_cv: int = 3

    # Decision policy (for reporting)
    decision_policy: Literal["topk", "f1"] = "topk"
    topk_k: int = 50  # FINAL DEFAULT

    def __post_init__(self) -> None:
        # Default columns for AI4I2020
        if self.cat_cols is None:
            object.__setattr__(self, "cat_cols", ["Type"])

        if self.num_cols is None:
            object.__setattr__(
                self,
                "num_cols",
                [
                    "Air temperature [K]",
                    "Process temperature [K]",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]",
                    "TempDiff",
                    "Torque_x_RPM",
                ],
            )

        if self.drop_cols is None:
            object.__setattr__(
                self,
                "drop_cols",
                [
                    "UDI",
                    "Product ID",
                    "TWF",
                    "HDF",
                    "PWF",
                    "OSF",
                    "RNF",
                    self.target,
                ],
            )
