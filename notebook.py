# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Prepare the next team's dataset
# MAGIC Follow CHALLENGE.md. These headings are a starting point; implement and verify each stage yourself.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 1 AS connection_check

# COMMAND ----------

from pathlib import Path
import json

# Copy this Git folder's absolute path from its Workspace menu; include /Workspace/.
repo_root = Path("/Workspace/Users/jhuma.chakraborty.irs@catalyte.ai/to6-databricks-challenge-starter/")
assert (repo_root / "README.md").is_file(), "Check the Git folder path"
practice_result = spark.sql("SELECT 1 AS connection_check").first().asDict()
assert practice_result == {"connection_check": 1}
(repo_root / "outputs").mkdir(exist_ok=True)
(repo_root / "outputs/connection-practice.json").write_text(
    json.dumps(practice_result) + "\n", encoding="utf-8"
)

# COMMAND ----------

import hashlib

INPUT_SCHEMA = "training.to6_challenge3_inputs"
INITIAL_TABLE = f"{INPUT_SCHEMA}.returns_initial"
CORRECTED_TABLE = f"{INPUT_SCHEMA}.returns_corrected"
LINEAGE_VIEW = f"{INPUT_SCHEMA}.source_lineage"
workspace_user = spark.sql("SELECT current_user() AS username").first()["username"]
OUTPUT_SCHEMA = "training.to6_c3_" + hashlib.sha256(workspace_user.encode("utf-8")).hexdigest()[:12]
print({"input_schema": INPUT_SCHEMA, "output_schema": OUTPUT_SCHEMA})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understand the input
# MAGIC `Record` Catalog descriptions, sample observations and the supplied lineage source.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE `training`.`to6_challenge3_inputs`.`returns_initial`;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE training.to6_challenge3_inputs.returns_corrected

# COMMAND ----------

# --- 1. Catalog descriptions ---
print("=== returns_initial schema ===")
spark.sql("DESCRIBE training.to6_challenge3_inputs.returns_initial").show(truncate=False)

print("\n=== returns_corrected schema ===")
spark.sql("DESCRIBE training.to6_challenge3_inputs.returns_corrected").show(truncate=False)

# --- 2. Sample observations ---
print("\n=== returns_initial sample ===")
spark.table("training.to6_challenge3_inputs.returns_initial").show(10, truncate=False)

print("\n=== returns_corrected sample ===")
spark.table("training.to6_challenge3_inputs.returns_corrected").show(10, truncate=False)

# --- 3. Source lineage ---
print("\n=== source_lineage ===")
spark.table("training.to6_challenge3_inputs.source_lineage").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Decide which batch is usable
# MAGIC Keep the initial rejection evidence and apply the same checks to the replacement.

# COMMAND ----------

# DBTITLE 1,Validate batches and reject initial
from pyspark.sql import functions as F, Window

# ---------- Data-contract rules (CHALLENGE.md) ----------
# Completeness: return_id, tax_year, income, deduction are required.
# Uniqueness:   each return_id occurs exactly once.
# Validity:     amounts are nonnegative (zero income is valid).
REQUIRED_FIELDS = ["return_id", "tax_year", "income", "deduction"]
AMOUNT_FIELDS = ["income", "deduction"]

def validate_batch(table_name):
    """Apply the same rules to a batch; return every affected record with its reasons."""
    df = spark.table(table_name)
    flagged = df.withColumn("_id_count", F.count(F.lit(1)).over(Window.partitionBy("return_id")))
    reasons = [F.when(F.col(c).isNull(), F.lit(f"completeness: required field {c} is null")) for c in REQUIRED_FIELDS]
    reasons.append(F.when(F.col("return_id").isNotNull() & (F.col("_id_count") > 1),
                          F.concat(F.lit("uniqueness: return_id occurs "), F.col("_id_count").cast("string"), F.lit(" times"))))
    reasons += [F.when(F.col(c) < 0, F.lit(f"validity: {c} is negative")) for c in AMOUNT_FIELDS]
    flagged = flagged.withColumn("rejection_reasons", F.filter(F.array(*reasons), lambda r: r.isNotNull())).drop("_id_count")
    affected = flagged.filter(F.size("rejection_reasons") > 0).orderBy("return_id")
    affected_rows = [r.asDict() for r in affected.collect()]
    return {
        "table": table_name,
        "total_rows": df.count(),
        "distinct_return_ids": df.select("return_id").distinct().count(),
        "affected_row_count": len(affected_rows),
        "affected_records": affected_rows,
        "passed": len(affected_rows) == 0,
    }

# ---------- 1. Initial batch: must be rejected before any feature output ----------
initial_result = validate_batch(INITIAL_TABLE)
print(f"=== {INITIAL_TABLE} ===")
print(f"rows={initial_result['total_rows']}, distinct return_ids={initial_result['distinct_return_ids']}, passed={initial_result['passed']}")
for r in initial_result["affected_records"]:
    print("  REJECTED:", r)

if not initial_result["passed"]:
    # Preserve affected records + reasons as a table and as a small export
    REJECTED_TABLE = f"{OUTPUT_SCHEMA}.initial_rejected_records"
    spark.createDataFrame(initial_result["affected_records"]).select(
        "return_id", "taxpayer_id", "tax_year", "income", "deduction", "rejection_reasons"
    ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(REJECTED_TABLE)
    rejection_evidence = {
        "decision": "REJECTED",
        "rules": {"completeness": REQUIRED_FIELDS, "uniqueness": "return_id exactly once", "validity": "amounts >= 0"},
        **initial_result,
        "next_action": f"Do not write features from {INITIAL_TABLE}; assess {CORRECTED_TABLE} with the same rules.",
    }
    (repo_root / "outputs/initial-rejected-records.json").write_text(json.dumps(rejection_evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Initial batch REJECTED; {initial_result['affected_row_count']} affected records saved to {REJECTED_TABLE} and outputs/initial-rejected-records.json")

# ---------- 2. Replacement batch: same rules ----------
corrected_result = validate_batch(CORRECTED_TABLE)
print(f"\n=== {CORRECTED_TABLE} ===")
print(f"rows={corrected_result['total_rows']}, distinct return_ids={corrected_result['distinct_return_ids']}, passed={corrected_result['passed']}")
for r in corrected_result["affected_records"]:
    print("  REJECTED:", r)
(repo_root / "outputs/corrected-batch-assessment.json").write_text(json.dumps(corrected_result, indent=2) + "\n", encoding="utf-8")

# ---------- 3. Decision: only the corrected batch may feed features ----------
if not corrected_result["passed"]:
    raise ValueError(f"{CORRECTED_TABLE} also fails the data contract; stop and return it to the data owner.")
ACCEPTED_TABLE = CORRECTED_TABLE
print(f"\nACCEPTED batch for the final run: {ACCEPTED_TABLE}")

# ---------- (previous exploratory checks retained below) ----------
# ---------- 1. Load both batches ----------
initial_df = spark.table(INITIAL_TABLE)
corrected_df = spark.table(CORRECTED_TABLE)

# ---------- 2. Define quality checks (same for both batches) ----------
def run_quality_checks(df, label):
    """Run a standard set of quality checks and return a summary dict."""
    total_rows = df.count()
    results = {"label": label, "total_rows": total_rows}

    # Check 1: Null primary-key / critical columns
    null_counts = df.select(
        *[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
    ).first().asDict()
    results["null_counts"] = {k: v for k, v in null_counts.items() if v and v > 0}

    # Check 2: Duplicate rows (full-row duplicates)
    dup_count = total_rows - df.dropDuplicates().count()
    results["duplicate_rows"] = dup_count

    # Check 3: Row count sanity
    results["is_empty"] = total_rows == 0

    # Check 4: Schema snapshot
    results["columns"] = df.columns
    results["dtypes"] = {f.name: str(f.dataType) for f in df.schema.fields}

    return results

# ---------- 3. Apply checks to the INITIAL batch (rejection evidence) ----------
initial_checks = run_quality_checks(initial_df, "returns_initial")
print("=== INITIAL batch quality checks ===")
for k, v in initial_checks.items():
    print(f"  {k}: {v}")

# ---------- 4. Apply the SAME checks to the CORRECTED (replacement) batch ----------
corrected_checks = run_quality_checks(corrected_df, "returns_corrected")
print("\n=== CORRECTED batch quality checks ===")
for k, v in corrected_checks.items():
    print(f"  {k}: {v}")

# ---------- 5. Decision ----------
initial_has_issues = bool(initial_checks["null_counts"]) or initial_checks["duplicate_rows"] > 0 or initial_checks["is_empty"]
corrected_has_issues = bool(corrected_checks["null_counts"]) or corrected_checks["duplicate_rows"] > 0 or corrected_checks["is_empty"]

print("\n=== DECISION ===")
print(f"  Initial batch has issues:   {initial_has_issues}")
print(f"  Corrected batch has issues: {corrected_has_issues}")

# The rule-based decision above (ACCEPTED_TABLE) is authoritative; no fallback to the initial batch.
assert initial_has_issues and not corrected_has_issues

# COMMAND ----------

# DBTITLE 1,Rejection explanation
# MAGIC %md
# MAGIC **Why the initial batch is rejected and what happens next**
# MAGIC
# MAGIC Rules (from CHALLENGE.md): *completeness*: `return_id`, `tax_year`, `income`, `deduction` must be non-null; *uniqueness*: each `return_id` occurs exactly once; *validity*: amounts are nonnegative.
# MAGIC
# MAGIC Affected records in `returns_initial` (13 rows, 12 distinct IDs), preserved in `initial_rejected_records` and `outputs/initial-rejected-records.json`:
# MAGIC * `R02` (2022, deduction 33): `income` is NULL, so it fails **completeness**. The amount is unknown, so it is not replaced with zero.
# MAGIC * `R01` (2022, income 100, deduction 20) appears **twice**, so both rows fail **uniqueness**. We don't drop either row silently.
# MAGIC
# MAGIC Next action: write no features from the initial batch. Re-assess `returns_corrected` with the same `validate_batch` rules. It has 12 rows and 12 distinct IDs, `R02` now has income 200, `R01` occurs once, and there are no violations. It is used for the final run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Deliver a checked feature table
# MAGIC Implement the task's data contract and verify the saved result by reading it back.

# COMMAND ----------

# DBTITLE 1,Build and verify return_features
from pyspark.sql import functions as F
import json

# ---------- 0. Gate: features only from the batch that passed the rules ----------
assert not initial_result["passed"], "Initial batch must be rejected before feature output is written."
assert corrected_result["passed"] and ACCEPTED_TABLE == CORRECTED_TABLE
accepted_df = spark.table(ACCEPTED_TABLE)

# ---------- 1. Feature contract ----------
# deduction_to_income = deduction / income; NULL when income = 0 (row kept).
# taxpayer_id, income, deduction are excluded (need-to-know).
FEATURE_COLUMNS = ["return_id", "tax_year", "deduction_to_income"]
features_df = accepted_df.select(
    "return_id",
    "tax_year",
    F.when(F.col("income") == 0, F.lit(None).cast("double"))
     .otherwise(F.col("deduction").cast("double") / F.col("income").cast("double"))
     .alias("deduction_to_income"),
)

FEATURES_TABLE = f"{OUTPUT_SCHEMA}.return_features"
features_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FEATURES_TABLE)

# ---------- 2. Verify by reading the saved table back ----------
saved = spark.table(FEATURES_TABLE)
src_count = accepted_df.count()
assert saved.columns == FEATURE_COLUMNS, saved.columns
assert saved.count() == src_count, "every accepted return must be present"
assert saved.select("return_id").distinct().count() == src_count, "each return exactly once"
assert saved.select("return_id").exceptAll(accepted_df.select("return_id")).count() == 0
assert accepted_df.select("return_id").exceptAll(saved.select("return_id")).count() == 0

# Nonzero case: expected value derived directly from the source row
src_row = accepted_df.filter("income > 0").orderBy("return_id").first()
expected_ratio = src_row["deduction"] / src_row["income"]
saved_ratio = saved.filter(F.col("return_id") == src_row["return_id"]).first()["deduction_to_income"]
assert abs(saved_ratio - expected_ratio) < 1e-12, (saved_ratio, expected_ratio)
print(f"Nonzero check: {src_row['return_id']} deduction {src_row['deduction']} / income {src_row['income']} "
      f"= {expected_ratio}; saved = {saved_ratio}")

# Zero-income cases: present once with NULL ratio
zero_ids = sorted(r["return_id"] for r in accepted_df.filter("income = 0").select("return_id").collect())
zero_saved = {r["return_id"]: r["deduction_to_income"] for r in saved.filter(F.col("return_id").isin(zero_ids)).collect()}
assert zero_ids and sorted(zero_saved) == zero_ids and all(v is None for v in zero_saved.values())
print(f"Zero-income check: {zero_ids} present with NULL ratio")

# Every nonzero ratio matches the source exactly
mismatches = (saved.alias("s").join(accepted_df.alias("a"), "return_id")
              .filter("a.income > 0 AND abs(s.deduction_to_income - a.deduction / a.income) > 1e-12").count())
assert mismatches == 0

feature_rows = [r.asDict() for r in saved.orderBy("return_id").collect()]
display(saved.orderBy("return_id"))

# ---------- 3. Export features + verification evidence ----------
(repo_root / "outputs/return_features.json").write_text(json.dumps(feature_rows, indent=2) + "\n", encoding="utf-8")
feature_verification = {
    "feature_table": FEATURES_TABLE,
    "source_table": ACCEPTED_TABLE,
    "columns": saved.columns,
    "row_count": saved.count(),
    "distinct_return_ids": saved.select("return_id").distinct().count(),
    "nonzero_check": {"return_id": src_row["return_id"], "source_deduction": src_row["deduction"],
                      "source_income": src_row["income"], "expected": expected_ratio, "saved": saved_ratio},
    "zero_income_check": {"return_ids": zero_ids, "saved_ratios": [zero_saved[i] for i in zero_ids]},
    "nonzero_ratio_mismatches": mismatches,
}
(repo_root / "outputs/return-features-verification.json").write_text(json.dumps(feature_verification, indent=2) + "\n", encoding="utf-8")
print(json.dumps(feature_verification, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Make the handoff usable
# MAGIC Save the required year tables. Keep references and IDs-and-values evidence for two successful runs.
# MAGIC Write SUBMISSION.md and the small exports inside this Databricks Git folder.

# COMMAND ----------

# DBTITLE 1,Save splits and record run evidence
from pyspark.sql import functions as F
from datetime import datetime, timezone
import json, hashlib

# ---------- 1. Year splits from the SAVED feature table ----------
SPLITS = {"features_train": 2022, "features_validation": 2023, "features_test": 2024}
features = spark.table(FEATURES_TABLE)
unassigned = features.filter(~F.col("tax_year").isin(list(SPLITS.values())) | F.col("tax_year").isNull()).count()
assert unassigned == 0, f"{unassigned} accepted returns have no assigned split year"

for name, year in SPLITS.items():
    (features.filter(F.col("tax_year") == year)
     .write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{OUTPUT_SCHEMA}.{name}"))

# ---------- 2. Read back and check coverage ----------
def table_rows(full_name):
    return [r.asDict() for r in spark.table(full_name).orderBy("return_id").collect()]

snapshot = {"return_features": table_rows(FEATURES_TABLE)}
for name, year in SPLITS.items():
    rows = table_rows(f"{OUTPUT_SCHEMA}.{name}")
    assert spark.table(f"{OUTPUT_SCHEMA}.{name}").columns == FEATURE_COLUMNS
    assert rows and all(r["tax_year"] == year for r in rows), f"{name} contains a wrong year"
    snapshot[name] = rows
    (repo_root / f"outputs/{name}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"{OUTPUT_SCHEMA}.{name} ({year}): {[r['return_id'] for r in rows]}")

split_ids = [r["return_id"] for name in SPLITS for r in snapshot[name]]
feature_ids = [r["return_id"] for r in snapshot["return_features"]]
assert len(split_ids) == len(set(split_ids)), "a return appears in more than one split"
assert sorted(split_ids) == sorted(feature_ids), "split union differs from return_features"
# Split rows carry the same values as return_features
assert sorted([r for n in SPLITS for r in snapshot[n]], key=lambda r: r["return_id"]) == snapshot["return_features"]
print(f"Coverage check: {len(feature_ids)} accepted returns, each in exactly one split, values unchanged")

# ---------- 3. Record this run and compare with the previous run ----------
snapshot_sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_path = ctx.notebookPath().get()
try:
    from databricks.sdk import WorkspaceClient
    nb_id = WorkspaceClient().workspace.get_status(notebook_path).object_id
    notebook_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/editor/notebooks/{nb_id}"
except Exception as e:
    notebook_url = f"unresolved ({type(e).__name__}); notebook path {notebook_path}"

run_record = {
    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    "notebook_url": notebook_url,
    "notebook_path": notebook_path,
    "initial_batch": {"table": INITIAL_TABLE, "decision": "REJECTED",
                      "affected_return_ids": [r["return_id"] for r in initial_result["affected_records"]]},
    "accepted_batch": ACCEPTED_TABLE,
    "row_counts": {k: len(v) for k, v in snapshot.items()},
    "snapshot_sha256": snapshot_sha,
    "snapshot": snapshot,
}

evidence_path = repo_root / "outputs/run-comparison.json"
prior_runs = json.loads(evidence_path.read_text())["runs"] if evidence_path.exists() else []
runs = (prior_runs + [run_record])[-2:]   # keep the two most recent complete runs
comparison = None
if len(runs) == 2:
    comparison = {
        "runs_compared": [r["executed_at_utc"] for r in runs],
        "identical_ids_and_values": runs[0]["snapshot"] == runs[1]["snapshot"],
        "identical_sha256": runs[0]["snapshot_sha256"] == runs[1]["snapshot_sha256"],
        "row_counts": [r["row_counts"] for r in runs],
    }
evidence_path.write_text(json.dumps({"runs": runs, "comparison": comparison}, indent=2) + "\n", encoding="utf-8")
print(f"Run recorded at {run_record['executed_at_utc']} | snapshot sha256 {snapshot_sha}")
print("Comparison with previous run:", comparison)
if comparison:
    assert comparison["identical_ids_and_values"], "IDs or values differ between the two runs"

# COMMAND ----------

# DBTITLE 1,Write SUBMISSION.md handoff
# ---------- Handoff: SUBMISSION.md built from this run's actual results ----------
rejected_lines = "\n".join(
    f"| {r['return_id']} | {r['tax_year']} | {r['income']} | {r['deduction']} | {'; '.join(r['rejection_reasons'])} |"
    for r in initial_result["affected_records"]
)
split_lines = "\n".join(
    f"| `{OUTPUT_SCHEMA}.{n}` | {y} | {len(snapshot[n])} | {', '.join(r['return_id'] for r in snapshot[n])} |"
    for n, y in SPLITS.items()
)
run_lines = "\n".join(f"| {i+1} | {r['executed_at_utc']} | {r['notebook_url']} | `{r['snapshot_sha256'][:16]}...` |"
                      for i, r in enumerate(runs))
if comparison:
    comparison_text = (f"The two runs left the same return IDs and values: **{comparison['identical_ids_and_values']}** "
                       f"(full snapshots and SHA-256 compared; row counts {comparison['row_counts'][-1]}).")
else:
    comparison_text = "Only one run is recorded so far. Re-run the whole notebook to record the second run and compare."
nz = feature_verification["nonzero_check"]

submission_md = f"""# SUBMISSION: Challenge 3 case-selection dataset

## Inputs
* Initial batch (rejected): `{INITIAL_TABLE}`
* Replacement batch (accepted): `{CORRECTED_TABLE}`
* Lineage view: `{LINEAGE_VIEW}`. Both deliveries come from `training.to6_challenge3_inputs.parsed_source`.
* Row meaning: one row = one return; `return_id` is the unique identifier; `taxpayer_id` is the filer.

## Outputs (schema `{OUTPUT_SCHEMA}`)
* `{FEATURES_TABLE}` with columns `return_id`, `tax_year`, `deduction_to_income` ({feature_verification['row_count']} rows)
* `{OUTPUT_SCHEMA}.initial_rejected_records`: the affected initial records and their rejection reasons

| Split table | Year | Rows | Return IDs |
|---|---|---|---|
{split_lines}

## Ratio definition
`deduction_to_income = deduction / income` (double). When `income = 0`, the return stays in the table with a NULL ratio.
`taxpayer_id`, `income` and `deduction` are excluded from the outputs (need-to-know rule).

## Checks and results
The same rules are applied to both batches: **completeness** (`return_id`, `tax_year`, `income` and `deduction` are non-null),
**uniqueness** (each `return_id` occurs once) and **validity** (amounts >= 0).

Initial batch: {initial_result['total_rows']} rows, {initial_result['distinct_return_ids']} distinct IDs, so it is **REJECTED** before any feature output is written.

| return_id | tax_year | income | deduction | reason |
|---|---|---|---|---|
{rejected_lines}

Next action taken: no features from the initial batch. Assessed `{CORRECTED_TABLE}` with the same rules:
{corrected_result['total_rows']} rows, {corrected_result['distinct_return_ids']} distinct IDs, {corrected_result['affected_row_count']} violations, so it is **ACCEPTED**.
No unknown amounts were replaced and no records were discarded.

Readback of `return_features`:
* Exact columns; every accepted return appears exactly once; same ID set as the source.
* Nonzero check: {nz['return_id']}, {nz['source_deduction']} / {nz['source_income']} = {nz['expected']}; saved value = {nz['saved']}.
* Zero-income check: {feature_verification['zero_income_check']['return_ids']} present with NULL ratio.
* All nonzero ratios match the source (mismatches: {feature_verification['nonzero_ratio_mismatches']}).

Splits: each split contains only its assigned year; every accepted return is in exactly one split; values match `return_features`.
All tables are written in overwrite mode, so re-runs replace the data instead of appending copies.

## Run references
| Run | Executed (UTC) | Notebook URL | Snapshot SHA-256 |
|---|---|---|---|
{run_lines}

{comparison_text} Evidence: `outputs/run-comparison.json`.

## Exports (`outputs/`)
* `initial-rejected-records.json`: affected initial records, rules, reasons and next action
* `corrected-batch-assessment.json`: the same rules applied to the replacement batch
* `return_features.json`, `return-features-verification.json`
* `features_train.json`, `features_validation.json`, `features_test.json`
* `run-comparison.json`: IDs and values from both runs, plus the comparison
* `connection-practice.json` (ungraded practice)

## Not verified
* The upstream `parsed_source` was not reconciled row by row. Lineage comes from the supplied view, which says it does not claim a Catalog UI check.
* No check covers model leakage or labels; later modelling work must assess what is available at prediction time.
* The GitHub commit contents must be checked after push; the commit SHA is reported in Classroom, not here.
"""
(repo_root / "SUBMISSION.md").write_text(submission_md, encoding="utf-8")
print(submission_md)