# Detection Spec

## Scope

This repo uses detection for conference-style empirical research.

Must optimize for:
- correctness,
- comparability across methods,
- reproducibility,
- explicit deviation tracking.

Not a production security system.

## Active IDS Assumption

For the IDS datasets in scope:
- preprocess to `all-numeric` features,
- drop identifier / leakage-prone columns,
- encode labels separately,
- do not block baseline detection on mixed categorical handling.

Mixed `x_num + x_cat` support may remain in models, but it is not the default IDS detection path.

## Detection Tracks

### A. Sample-Level

Goal:
- score and rank suspicious poisoned samples.

Typical method:
- `Spectral Signatures`

Primary metrics:
- `precision`
- `recall`
- `f1`
- `top-k recall`
- optional `auroc`, `average_precision`

### B. Model-Level / Class-Level

Goal:
- decide whether a trained model is infected,
- identify likely target class,
- optionally infer source class or trigger-like artifact.

Typical methods:
- `Neural Cleanse`
- `MM-BD`
- `MLBD`
- `CSO` variants

Primary metrics:
- infected-vs-clean accuracy
- target-class accuracy
- source-class accuracy if applicable
- false positive rate
- true positive rate
- runtime

Never merge Track A and Track B into one ambiguous metric table.

## Shared Input Contract

Every detector should receive a context with:
- `model`
- `model_name`
- `model_family`
- `num_classes`
- `device`
- `detection_split`
- `seed`

Optional:
- `clean_support_split`
- `poisoned_indices`
- `attack_target_label`
- `attack_source_labels`
- `feature_names`
- `feature_bounds`
- `class_names`
- `run_dir`
- `model_metadata`
- `attack_metadata`

Rules:
- `detection_split` must be recorded explicitly.
- `clean_support_split` is required only for methods that need clean labeled data.

## Shared Output Contract

Every detector should return one serializable result object with:
- `detector_name`
- `track_type`
- `status`
- `seed`
- `runtime_sec`
- `artifacts`
- `deviation_note`

For sample-level detectors:
- `sample_scores`
- `sample_ranking`
- `sample_flags`

For class-level detectors:
- `class_scores`
- `predicted_is_infected`
- `predicted_target_class`

Optional:
- `predicted_source_class`
- `estimated_trigger`
- `estimated_mask`
- `estimated_perturbation`
- `optimization_trace`
- `feature_layer_name`
- `threshold`

## Shared Artifact Contract

Each run should save:
- resolved config
- seed
- detector name
- model checkpoint reference
- data split reference
- raw scores
- thresholded outputs
- summary metrics
- runtime
- deviation note

Preferred formats:
- JSON: summary/config/trace
- CSV or parquet: score tables
- `.npy` or `.pt`: tensors

## Method Assumptions

### Spectral Signatures

Track:
- sample-level

Needs:
- trained model
- feature extractor
- candidate samples with class labels
- poison fraction estimate or removal budget

Local use:
- use `forward_features()`
- run on attacked candidate data
- evaluate with localization metrics
- current repo lock for IDS:
  - unknown-target assumption,
  - scan all observed classes,
  - save both `raw_sample_scores` and final `sample_scores`,
  - default `threshold_mode=ids_benchmark`,
  - keep `threshold_mode=paper` only for explicit reproduction runs

Limits:
- needs a candidate sample set; not a pure post-training detector

Paper:
- https://papers.nips.cc/paper_files/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf

Code:
- https://github.com/MadryLab/backdoor_data_poisoning

### Neural Cleanse

Track:
- model-level

Needs:
- trained model
- clean labeled support data
- trigger parameterization
- class-wise reverse engineering

Original assumption:
- image-like `mask + pattern` trigger

Local IDS assumption:
- numeric-only sparse additive perturbation + feature mask

Rule:
- treat numeric IDS version as a controlled deviation, not paper-faithful replication

Paper:
- https://bolunwang.github.io/assets/docs/backdoor-sp19.pdf

Code:
- https://github.com/bolunwang/backdoor

### MM-BD

Track:
- model-level

Needs:
- trained model
- logits access
- class-wise optimization of synthetic input

Original advantage:
- no clean support set required

Local IDS assumption:
- optimize in numeric feature space
- require valid feature bounds from preprocessing metadata by default
- only allow split-derived bounds as an explicit fallback
- use multiple random restarts
- current repo use:
  - no `clean_support_split`,
  - use `detection_split` only as a numeric reference for bounds when needed,
  - save `class_scores` and optimized per-class inputs,
  - use gamma p-value on the maximum-margin statistic vector

Paper:
- https://arxiv.org/abs/2205.06900

Code:
- https://github.com/wanghangpsu/MM-BD

### MLBD

Track:
- model-level

Needs:
- trained model
- logits access
- class-wise synthetic input optimization

Local IDS assumption:
- same search space and constraints as MM-BD

Use:
- simpler baseline before or alongside CSO

Reference:
- CSO paper baseline

### CSO

Track:
- add-on, not standalone

Needs:
- baseline detector already working
- clean support set per class
- internal feature representation layer

Local rule:
- use `forward_features()` or another explicitly documented layer
- record support-set size and feature layer

Paper:
- https://arxiv.org/pdf/2512.08129

## Recommended Order

For IDS numeric-only baseline:
1. `Spectral Signatures`
2. `MM-BD`
3. `MLBD`
4. `MM-BD-CSO`
5. `MLBD-CSO`
6. `Neural Cleanse`
7. `NC-CSO`

Reason:
- lowest adaptation risk first,
- strongest fit to numeric IDS baseline first,
- delay image-centric adaptation.

## Non-Negotiable Rules

- Shared metrics must be computed by shared code, not ad hoc in each detector.
- Threshold source must be recorded.
- Random restarts and seeds must be recorded.
- Every deviation from paper or official code must be written down.
- A detector is not "ready" until its outputs, metrics, and artifacts are reproducible.

See [BASE_DETECTOR_BLUEPRINT.md](./BASE_DETECTOR_BLUEPRINT.md).
