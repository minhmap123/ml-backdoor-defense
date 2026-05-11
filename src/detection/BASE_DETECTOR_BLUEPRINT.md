# Base Detector Blueprint

## Goal

Keep one shared detector contract for model-level backdoor detection.

The benchmark no longer evaluates poisoned-sample localization. A detector should decide whether the model is infected, identify the suspicious target class, and expose the score-threshold comparison used for that decision.

## File Layout

```text
src/detection/
  README.md
  BASE_DETECTOR_BLUEPRINT.md
  __init__.py
  base.py
  types.py
  utils.py
  mlbd.py
  mlbd_cso.py
  mm_bd.py
  mmbd_cso.py
  neural_cleanse.py
  nc_cso.py
  cso.py
  trigger_report.py
```

## Core Types

`DetectorContext` should carry the trained model, relevant data splits, model/data metadata, true attack target when available, detector config, seed, device, and run directory.

`DetectorResult` should carry:
- class-level scores,
- infected/not-infected prediction,
- predicted target/source class when the detector crosses threshold,
- candidate target class even when threshold is not crossed,
- candidate target score,
- decision score,
- decision threshold,
- decision margin,
- threshold direction,
- artifacts and deviation notes.

## Base Class

`BaseDetector.run(...)` owns:
- context validation,
- seed setup,
- shared class-level metric enrichment,
- artifact saving,
- runtime/status logging.

Detector subclasses own only method-specific logic in `_run_impl(...)`.

## Decision Reporting

Every class-level detector should populate:
- `candidate_target_class`
- `candidate_target_score`
- `decision_score`
- `decision_threshold`
- `decision_greater_is_infected`

Shared code derives:
- `decision_margin = decision_score - threshold` when larger means infected,
- `decision_margin = threshold - decision_score` when smaller means infected.

Positive margin means the detector crossed its configured threshold.

## Artifacts

Recommended per-run layout:

```text
results/.../detection/
  summary.json
  class_scores.csv
  class_details.csv
  optimization_trace.json
  estimated_pattern.npy
  estimated_trigger.npy
  estimated_mask.npy
```

Absent artifacts are fine; fake placeholders are not.

## Active Methods

- Neural Cleanse / NC-CSO use MAD over recovered mask norms.
- MM-BD / MMBD-CSO use a gamma-tail decision over class scores.
- MLBD / MLBD-CSO reuse the MM-BD scaffold with target-logit objective.
