# ============================================================
# Notebook  : 02_silver_clean
# Layer     : Silver
# Purpose   : Read Bronze → enforce schema → data quality
#             checks → write Delta silver table
# Runs on   : Databricks Community Edition (DBR 14.x)
# Pipeline  : 01 → 02 → 03 → 04
# Depends on: 01_bronze_ingest must run first
# ============================================================

from pyspark.sql import functions as F

# ── Config ───────────────────────────────────────────────────────
SCHEMA       = "ai4i2020_demo_db"
BRONZE_TABLE = f"{SCHEMA}.ai4i_bronze_raw"
SILVER_TABLE = f"{SCHEMA}.ai4i_silver_clean"
EXPECTED_ROWS = 10_000

spark.sql(f"USE {SCHEMA}")

# ── Read Bronze ──────────────────────────────────────────────────
bronze = spark.table(BRONZE_TABLE)
print(f"[Silver] Bronze rows={bronze.count()}  cols={bronze.columns}")

# ── Enforce schema ───────────────────────────────────────────────
# Cast all columns to correct types.
# Bronze stores everything as strings/longs from CSV — silver enforces types.
silver = bronze.select(
    F.col("udi").cast("int").alias("udi"),
    F.col("product_id").cast("string").alias("product_id"),
    F.col("type").cast("string").alias("type"),
    F.col("air_temperature_k").cast("double").alias("air_temperature_k"),
    F.col("process_temperature_k").cast("double").alias("process_temperature_k"),
    F.col("rotational_speed_rpm").cast("int").alias("rotational_speed_rpm"),
    F.col("torque_nm").cast("double").alias("torque_nm"),
    F.col("tool_wear_min").cast("int").alias("tool_wear_min"),
    F.col("machine_failure").cast("int").alias("machine_failure"),
    # Failure mode flags (binary)
    F.col("twf").cast("int").alias("twf"),   # Tool Wear Failure
    F.col("hdf").cast("int").alias("hdf"),   # Heat Dissipation Failure
    F.col("pwf").cast("int").alias("pwf"),   # Power Failure
    F.col("osf").cast("int").alias("osf"),   # Overstrain Failure
    F.col("rnf").cast("int").alias("rnf"),   # Random Failure
)

# ── Data Quality Checks ──────────────────────────────────────────
rows = silver.count()
print(f"[Silver] rows={rows}")

# 1. Row count
if rows != EXPECTED_ROWS:
    raise ValueError(f"[Silver] Unexpected row count {rows} (expected {EXPECTED_ROWS})")

# 2. Null checks on critical columns
critical = ["udi", "type", "machine_failure"]
nulls = {c: silver.filter(F.col(c).isNull()).count() for c in critical}
print(f"[Silver] Null counts in critical columns: {nulls}")
if any(v != 0 for v in nulls.values()):
    raise ValueError(f"[Silver] Nulls found in critical columns: {nulls}")

# 3. Duplicate primary key check
dupes = silver.groupBy("udi").count().filter(F.col("count") > 1).count()
print(f"[Silver] Duplicate UDI count: {dupes}")
if dupes != 0:
    raise ValueError(f"[Silver] {dupes} duplicate UDI(s) detected — data integrity issue")

# 4. Label distribution — must have both classes
label_dist = {
    r["machine_failure"]: r["count"]
    for r in silver.groupBy("machine_failure").count().collect()
}
print(f"[Silver] Label distribution: {label_dist}")
if 0 not in label_dist or 1 not in label_dist:
    raise ValueError(f"[Silver] Missing class in machine_failure: {label_dist}")

# 5. Sensor range sanity checks (domain knowledge)
temp_min = silver.agg(F.min("air_temperature_k")).collect()[0][0]
temp_max = silver.agg(F.max("air_temperature_k")).collect()[0][0]
print(f"[Silver] Air temperature range: {temp_min:.1f} – {temp_max:.1f} K")
if temp_min < 200 or temp_max > 400:
    print(f"[Silver] ⚠ Air temperature out of expected range (200–400 K)")

print(f"[Silver] ✅ All quality checks passed")

# ── Write Silver ─────────────────────────────────────────────────
(silver.write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(SILVER_TABLE))

written = spark.table(SILVER_TABLE).count()
print(f"[Silver] ✅ Wrote {SILVER_TABLE}  rows={written}")

if written != rows:
    raise ValueError(f"[Silver] Row count mismatch: ingested={rows}  written={written}")

print("[Silver] Done.")