# Detection Spec

## Scope

The active benchmark is model-level backdoor detection for numeric IDS tabular models.

Primary questions:
- is the trained model backdoored?
- which target class is most suspicious?
- what is that class score?
- what decision score is compared with the detector threshold?
- how far is the decision score from the threshold?

Spectral Signatures and poisoned-sample localization metrics are out of active scope.

## Active Assumption

For IDS datasets:
- inputs are all-numeric after preprocessing,
- feature bounds are model-input bounds from preprocessing metadata,
- categorical model paths may exist but are not the active detection baseline.

## Input Contract

`DetectorContext` provides:
- `model`
- `model_name`
- `model_family`
- `num_classes`
- `detection_split`
- `clean_support_split` when a detector needs clean support data
- `attack_target_label`
- `attack_source_labels`
- `feature_metadata`
- `model_metadata`
- `attack_metadata`
- `detector_cfg`
- `seed`
- `device`
- `run_dir`

## Output Contract

`DetectorResult` provides:
- `detector_name`
- `track_type`
- `status`
- `seed`
- `runtime_sec`
- `class_scores`
- `predicted_is_infected`
- `predicted_target_class`
- `predicted_source_class`
- `candidate_target_class`
- `candidate_target_score`
- `decision_score`
- `decision_threshold`
- `decision_margin`
- `decision_greater_is_infected`
- `thresholds`
- `artifacts`
- `deviation_note`

Optional trigger-recovery outputs:
- `estimated_trigger`
- `estimated_mask`
- `estimated_perturbation`
- `optimized_inputs`
- `optimization_trace`

## Metrics

Shared class/model-level metrics:
- `detection/is_infected_accuracy`
- `detection/target_class_accuracy`
- `detection/source_class_accuracy`
- `detection/false_positive_rate`
- `detection/true_positive_rate`
- `detection/candidate_target_class`
- `detection/candidate_target_score`
- `detection/decision_score`
- `detection/decision_threshold`
- `detection/decision_margin`
- `detection/runtime_sec`

`decision_margin > 0` means the detector decision crosses the configured threshold.

## Artifacts

Each detector run saves:
- `summary.json`
- `class_scores.csv` when class scores exist
- `class_details.csv` when per-class details exist
- `optimization_trace.json` when optimization trace exists
- recovered trigger/mask arrays when available

## Active Methods

- `Neural Cleanse`
- `NC-CSO`
- `MM-BD`
- `MMBD-CSO`
- `MLBD`
- `MLBD-CSO`
- `none` for controls

See [BASE_DETECTOR_BLUEPRINT.md](./BASE_DETECTOR_BLUEPRINT.md).
