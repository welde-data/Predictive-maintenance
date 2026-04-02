# ============================================================
# Notebook  : 01_bronze_ingest
# Layer     : Bronze
# Purpose   : Download AI4I 2020 CSV → sanitize columns
#             → write Delta bronze table (idempotent)
# Runs on   : Databricks Community Edition (DBR 14.x)
# Pipeline  : 01 → 02 → 03 → 04
# ============================================================

import re
import pandas as pd
from pyspark.sql import functions as F

# ── Config ───────────────────────────────────────────────────────
SCHEMA       = "ai4i2020_demo_db"
BRONZE_TABLE = f"{SCHEMA}.ai4i_bronze_raw"
SOURCE_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
EXPECTED_ROWS = 10_000

spark.sql(f"CREATE DATABASE IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE {SCHEMA}")

# ── Helpers ──────────────────────────────────────────────────────
def table_exists(full_name: str) -> bool:
    try:
        spark.table(full_name)
        return True
    except Exception:
        return False

def sanitize(col: str) -> str:
    """Lowercase + replace special chars with underscores (Delta safe)."""
    c = col.strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

# ── Idempotency check ────────────────────────────────────────────
if table_exists(BRONZE_TABLE):
    existing = spark.table(BRONZE_TABLE).count()
    print(f"[Bronze] Existing table found: rows={existing}")
    if existing == EXPECTED_ROWS:
        print("[Bronze] ✅ Row count matches. Skipping ingestion.")
    else:
        print(f"[Bronze] ⚠ Row count mismatch ({existing} ≠ {EXPECTED_ROWS}). Rebuilding...")
        spark.sql(f"DROP TABLE IF EXISTS {BRONZE_TABLE}")

# ── Ingest ───────────────────────────────────────────────────────
if not table_exists(BRONZE_TABLE):
    print(f"[Bronze] Downloading from: {SOURCE_URL}")
    pdf = pd.read_csv(SOURCE_URL)
    print(f"[Bronze] Downloaded rows={len(pdf)}  cols={list(pdf.columns)}")

    df_raw = spark.createDataFrame(pdf)

    # Sanitize column names (required for Delta compatibility)
    old_cols = df_raw.columns
    new_cols = [sanitize(c) for c in old_cols]

    if len(set(new_cols)) != len(new_cols):
        raise ValueError(f"[Bronze] Sanitized column names not unique: {new_cols}")

    df_bronze = df_raw
    for old, new in zip(old_cols, new_cols):
        df_bronze = df_bronze.withColumnRenamed(old, new)

    print(f"[Bronze] Sanitized columns: {df_bronze.columns}")

    # Validate required columns are present
    required = {
        "udi", "product_id", "type",
        "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
        "machine_failure", "twf", "hdf", "pwf", "osf", "rnf",
    }
    missing = required - set(df_bronze.columns)
    if missing:
        raise ValueError(f"[Bronze] Missing columns after sanitize: {missing}")

    # Write Bronze Delta table
    (df_bronze.write
        .mode("overwrite")
        .format("delta")
        .saveAsTable(BRONZE_TABLE))

    count = spark.table(BRONZE_TABLE).count()
    print(f"[Bronze] ✅ Wrote {BRONZE_TABLE}  rows={count}")

    if count != EXPECTED_ROWS:
        raise ValueError(f"[Bronze] Unexpected row count: {count} (expected {EXPECTED_ROWS})")

print("[Bronze] Done.")