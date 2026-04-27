# Base Detector Blueprint

## Goal

Build one shared detection layer before method-specific detectors.

Base layer must standardize:
- detector construction,
- input context,
- output result,
- metrics,
- artifacts,
- seeding,
- runtime reporting.

## Design Rules

1. Reproducibility first.
2. Shared contracts before method depth.
3. Minimal abstractions.
4. Explicit deviation logging.
5. One metric pipeline for all detectors.

Do not optimize for plugin complexity in v1.

## File Layout

```text
src/detection/
  README.md
  BASE_DETECTOR_BLUEPRINT.md
  __init__.py
  base.py
  types.py
  utils.py
  spectral_signatures.py
  mm_bd.py
  mlbd.py
  neural_cleanse.py
  cso.py
```

Keep it flat in the first iteration.

## Core Types

### `DetectorContext`

Should include:
- `model`
- `model_metadata`
- `detection_split`
- `clean_support_split`
- `poisoned_indices`
- `attack_metadata`
- `feature_metadata`
- `detector_cfg`
- `run_dir`
- `device`
- `seed`

`feature_metadata` should support:
- `feature_names`
- `feature_bounds_min`
- `feature_bounds_max`
- `num_numeric_features`
- `num_categorical_features`
- `cat_cardinalities`

For current IDS baseline:
- `num_categorical_features == 0` is expected.

### `DetectorResult`

Should include:
- `detector_name`
- `track_type`
- `status`
- `seed`
- `runtime_sec`
- `summary_metrics`
- `sample_scores`
- `sample_ranking`
- `sample_flags`
- `class_scores`
- `predicted_is_infected`
- `predicted_target_class`
- `predicted_source_class`
- `thresholds`
- `artifacts`
- `deviation_note`

Not every field is used by every detector, but the schema should stay stable.

### `ArtifactIndex`

Stable logical mapping, e.g.:
- `summary_json`
- `raw_scores_csv`
- `class_scores_csv`
- `optimization_trace_json`
- `estimated_pattern_npy`
- `plots`

## Base Class

### `BaseDetector`

Suggested methods:
- `__init__(cfg)`
- `run(context) -> DetectorResult`
- `_validate_context(context)`
- `_run_impl(context) -> DetectorResult`
- `_save_artifacts(result, context)`

Behavior:
- `run(...)` is the only public entrypoint
- `run(...)` handles seeding, timing, validation, artifact saving
- `_run_impl(...)` contains only detector-specific logic

## Registry

Provide:
- `get_detection(cfg)`

Responsibilities:
- resolve detector name
- instantiate detector
- fail loudly on unknown detector

Detector defaults belong in Hydra config, not in registry logic.

## Shared Modules

### `utils.py`

Include helpers such as:
- seed setup
- device resolution
- runtime measurement
- feature extraction
- logits extraction
- numeric input clamping
- ranking helpers
- sample-level precision/recall/F1
- top-k recall
- AUROC / AP when applicable
- infected-vs-clean accuracy
- target/source class accuracy
- false positive / true positive rate
- summary JSON writing
- score table writing
- optimization trace writing
- optional plot writing

Detector files should return raw scores; shared utility code should derive final summaries and save artifacts.

## Data Policy

Supported split roles:
- `detection_split`
- `clean_support_split`
- `evaluation_split`

Rules:
- `detection_split` is the actual detector input.
- `clean_support_split` is required for Neural Cleanse and CSO variants.
- `evaluation_split` is optional in v1 if metrics can be derived from detector outputs and attack metadata.

## Seed Policy

All detectors must respect one explicit seed.

If using randomness:
- restarts,
- support-set sampling,
- initialization,
- label subset sampling,

then derive from top-level seed and save the rule.

Suggested rule:
- restart seed = `seed + restart_id`

## Runtime Policy

Every run must record:
- wall-clock runtime,
- number of optimization steps,
- number of restarts,
- amount of data used

## Threshold Policy

Threshold source must be explicit:
- fixed from paper,
- outlier rule from paper,
- tuned on clean models,
- tuned on validation split

Always save:
- threshold value
- threshold source
- threshold tuning data if any

## Detector Integration Notes

### Spectral Signatures

Needs:
- feature extraction
- per-class grouping
- sample score table

First version:
- no automatic retraining after filtering
- only localization scores/ranking/metrics

### MM-BD / MLBD

Needs:
- class-wise optimization loop
- numeric feature bounds
- restart handling
- class score table
- anomaly detection over class scores

### Neural Cleanse

Needs:
- target-class loop
- perturbation/mask parameterization
- optimization trace
- anomaly detection over recovered trigger size

Implement only after base optimization plumbing is stable.

### CSO

Needs:
- baseline detector already working
- clean support set per class
- feature-layer extraction
- penalty integration into detector objective

Implement only after underlying detector is stable.

## Metrics Keys

Sample-level:
- `detection/precision`
- `detection/recall`
- `detection/f1`
- `detection/topk_recall`
- `detection/auroc`
- `detection/average_precision`
- `detection/runtime_sec`

Class/model-level:
- `detection/is_infected_accuracy`
- `detection/target_class_accuracy`
- `detection/source_class_accuracy`
- `detection/false_positive_rate`
- `detection/true_positive_rate`
- `detection/runtime_sec`

Auxiliary:
- `detection/num_candidates`
- `detection/num_classes_scored`
- `detection/num_restarts`
- `detection/num_optimization_steps`

## Artifact Layout

Recommended per-run layout:

```text
results/.../detection/
  summary.json
  config_resolved.yaml
  sample_scores.csv
  class_scores.csv
  optimization_trace.json
  estimated_pattern.npy
  notes.txt
```

Rules:
- `summary.json` must always exist
- absent artifacts are better than fake placeholders
- filenames should be stable across detectors where content type matches

`summary.json` should include:
- detector name
- resolved config
- seed
- model reference
- dataset reference
- metrics
- deviation note
- artifact index

## Hydra Configs

Expected configs:

```text
conf/detection/
  none.yaml
  spectral_signatures.yaml
  mm_bd.yaml
  mlbd.yaml
  neural_cleanse.yaml
  mm_bd_cso.yaml
  mlbd_cso.yaml
```

Each config should expose:
- `name`
- `seed`
- detector hyperparameters
- threshold settings
- restart settings
- runtime limits

## Implementation Order

### Phase 1

Create:
- `types.py`
- `base.py`
- `__init__.py`
- `utils.py`

Ready when:
- a dummy detector runs end-to-end
- result object saves correctly
- shared metrics/artifacts work

### Phase 2

Implement:
- `spectral_signatures.py`

Reason:
- best fit to current `forward_features()` contract
- lowest adaptation risk

### Phase 3

Implement:
- `mm_bd.py`
- `mlbd.py`

Reason:
- strong post-training fit for numeric IDS baseline
- no clean support required in original form

### Phase 4

Implement:
- `cso.py`
- `mm_bd_cso`
- `mlbd_cso`

### Phase 5

Implement:
- `neural_cleanse.py`

Highest adaptation risk; do last.

## Minimal Smoke Benchmark

Before multi-detector experiments, validate on:
- one dataset
- one model
- one attack
- one seed
- one checkpoint

Required path:
1. load model
2. build detector
3. run detector
4. compute shared metrics
5. save standard artifacts

## Definition of Ready

Base detector layer is ready when:
- detector builds from Hydra config
- execution is seed-stable
- runtime is measured
- scores save in standard format
- metrics come from shared code
- summaries are serializable
- assumptions and deviations are easy to inspect
