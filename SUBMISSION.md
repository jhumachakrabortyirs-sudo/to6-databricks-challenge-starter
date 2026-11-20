# SUBMISSION: Challenge 3 case-selection dataset

## Inputs
* Initial batch (rejected): `training.to6_challenge3_inputs.returns_initial`
* Replacement batch (accepted): `training.to6_challenge3_inputs.returns_corrected`
* Lineage view: `training.to6_challenge3_inputs.source_lineage`. Both deliveries come from `training.to6_challenge3_inputs.parsed_source`.
* Row meaning: one row = one return; `return_id` is the unique identifier; `taxpayer_id` is the filer.

## Outputs (schema `training.to6_c3_e5f099205cf8`)
* `training.to6_c3_e5f099205cf8.return_features` with columns `return_id`, `tax_year`, `deduction_to_income` (12 rows)
* `training.to6_c3_e5f099205cf8.initial_rejected_records`: the affected initial records and their rejection reasons

| Split table | Year | Rows | Return IDs |
|---|---|---|---|
| `training.to6_c3_e5f099205cf8.features_train` | 2022 | 4 | R01, R02, R03, R04 |
| `training.to6_c3_e5f099205cf8.features_validation` | 2023 | 4 | R05, R06, R07, R08 |
| `training.to6_c3_e5f099205cf8.features_test` | 2024 | 4 | R09, R10, R11, R12 |

## Ratio definition
`deduction_to_income = deduction / income` (double). When `income = 0`, the return stays in the table with a NULL ratio.
`taxpayer_id`, `income` and `deduction` are excluded from the outputs (need-to-know rule).

## Checks and results
The same rules are applied to both batches: **completeness** (`return_id`, `tax_year`, `income` and `deduction` are non-null),
**uniqueness** (each `return_id` occurs once) and **validity** (amounts >= 0).

Initial batch: 13 rows, 12 distinct IDs, so it is **REJECTED** before any feature output is written.

| return_id | tax_year | income | deduction | reason |
|---|---|---|---|---|
| R01 | 2022 | 100 | 20 | uniqueness: return_id occurs 2 times |
| R01 | 2022 | 100 | 20 | uniqueness: return_id occurs 2 times |
| R02 | 2022 | None | 33 | completeness: required field income is null |

Next action taken: no features from the initial batch. Assessed `training.to6_challenge3_inputs.returns_corrected` with the same rules:
12 rows, 12 distinct IDs, 0 violations, so it is **ACCEPTED**.
No unknown amounts were replaced and no records were discarded.

Readback of `return_features`:
* Exact columns; every accepted return appears exactly once; same ID set as the source.
* Nonzero check: R01, 20 / 100 = 0.2; saved value = 0.2.
* Zero-income check: ['R03', 'R09'] present with NULL ratio.
* All nonzero ratios match the source (mismatches: 0).

Splits: each split contains only its assigned year; every accepted return is in exactly one split; values match `return_features`.
All tables are written in overwrite mode, so re-runs replace the data instead of appending copies.

## Run references
| Run | Executed (UTC) | Notebook URL | Snapshot SHA-256 |
|---|---|---|---|
| 1 | 2026-11-20T17:01:14.599934+00:00 | https://dbc-48a16403-f66f.cloud.databricks.com/editor/notebooks/127024866559488 | `8fefc109b000748f...` |
| 2 | 2026-11-20T17:02:45.299136+00:00 | https://dbc-48a16403-f66f.cloud.databricks.com/editor/notebooks/127024866559488 | `8fefc109b000748f...` |

The two runs left the same return IDs and values: **True** (full snapshots and SHA-256 compared; row counts {'return_features': 12, 'features_train': 4, 'features_validation': 4, 'features_test': 4}). Evidence: `outputs/run-comparison.json`.

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
