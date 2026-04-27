# Data Format Specification

## Active Baseline Decision

For the current IDS research path in this repo, the active baseline assumption is:
- `IoTID20`, `CIC-IDS2017`, `CSE-CIC-IDS2018`, and `CIC-IoT-2023` should be preprocessed into `all-numeric` feature matrices.
- Raw identifier / leakage-prone columns such as flow IDs, IP addresses, ports, and timestamps should be dropped unless a specific experiment explicitly needs them.
- Labels remain separate and may be string labels in raw data, but must be encoded before model training.

Rationale:
- The current notebook-based preprocessing evidence for `IoTID20` already yields no remaining object columns in `X`.
- The CIC IDS CSV schemas used for ML are dominated by flow statistics and numeric protocol codes; after dropping identifier/timestamp columns and encoding labels, they naturally fit a numeric-only baseline.
- This keeps the baseline path aligned with the IDS use case and simplifies attack / detection / unlearning comparisons.

Scope note:
- Mixed `x_num + x_cat` support remains available in the model layer for future non-IDS tabular experiments or explicit ablations.
- It is not the default assumption for the IDS datasets currently in scope.

## Group 1: Flat-Input Models (MLP, ResNet)

**Return:** `(x_flat, y)` where x_flat = [x_num, x_cat_ohe]

**Shapes:**
- x_flat: (batch, n_num + sum(cat_cardinalities))
- y: (batch,)

**Processing:**
1. One-hot encode categorical features
2. Concatenate: [numerical, ohe_categorical]

---

## Group 2: Structured-Input Models (FT-Transformer, SAINT, TabNet)

**Return:** `((x_num, x_cat), y)` or dict `{"x_num": ..., "x_cat": ...}`

**Shapes:**
- x_num: (batch, n_num) float32
- x_cat: (batch, n_cat) int64
- y: (batch,)

**Constraint:** x_cat must be indices (0..cardinality-1), NOT one-hot

**Current IDS baseline override:**
- For the IDS datasets in scope, prefer `x_num` only and set `x_cat = null` / empty unless a dataset audit proves that meaningful categorical columns remain after preprocessing.
- In other words, structured-input models should be able to run in `numeric-only` mode as the default IDS baseline.

**Structured-model metadata requirement:**
- Dataset/preprocessing code must expose metadata for model construction, not just batch tensors.
- Required metadata for `FT-Transformer`, `SAINT`, and `TabNet`:
  - `num_numeric_features`
  - `cat_cardinalities`
  - `d_in` after preprocessing / feature selection
- This metadata is dataset-dependent and must be injected into `model_cfg` before calling `get_model(...)`.

**TabNet-specific note:**
- Even though TabNet internally consumes a single matrix input, the local wrapper expects structured raw tabular inputs first: `x_num` plus categorical indices `x_cat`.
- The wrapper reconstructs the raw mixed feature matrix and derives `cat_idxs` / `cat_dims` for the official DreamQuark implementation.
- Do **not** one-hot encode categoricals for TabNet in this repo.

**Current research assumption for SAINT:** fully observed features.
- The local SAINT wrapper uses the author-style embedding path, but currently assumes `cat_mask = 1` and `con_mask = 1` for all features.
- Therefore, the active pipeline does **not** pass dataset-level missingness masks into SAINT.
- For conference experiments in this repo, SAINT should be used only with datasets that have already had missing values removed or imputed during preprocessing.
- If true missing-feature semantics are needed later, the data pipeline must be extended to return `cat_mask` and `con_mask` explicitly.

---

## Cardinalities

Store per dataset: `cat_cardinalities = [c1, c2, ...]` (max_value + 1 for each categorical)

For the active IDS baseline path, `cat_cardinalities` is expected to be empty for most or all runs.

---

## Datasets in Scope
- IoTID20
- CIC-IDS2017
- CIC-IDS2018
- CIC-IoT-2023

**Checklist:**
- [ ] Audit each dataset after cleaning to confirm whether any meaningful categorical feature columns remain
- [ ] Default IDS preprocessing to numeric-only features unless the audit says otherwise
- [ ] Store cardinalities only if a dataset genuinely retains categorical features after preprocessing
- [ ] Store structured-model metadata needed for model construction
- [ ] Normalize numerical (StandardScaler/MinMaxScaler)
- [ ] Remove/impute missing values
- [ ] Train/val/test split (seed-based)
- [ ] Return correct format per model group
- [ ] For current SAINT baseline, keep features fully observed after preprocessing

---

## BaseTabularModel.parse_input() Auto-Converts

Models accept: tensor, dict `{"x": ...}` / `{"x_num": ..., "x_cat": ...}`, tuple `(x, y)`, ParsedModelInput

Data loaders can use any format; models normalize automatically.
