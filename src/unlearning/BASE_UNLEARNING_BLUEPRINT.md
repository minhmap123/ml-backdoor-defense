# Unlearning Blueprint

## Purpose

Build `src/unlearning` for the backdoor benchmark.

First support only:
- `noop`
- `oracle_retrain`
- `detected_retrain`

Do not add paper methods yet.

## Repo Facts

Assume:
- `run.py` calls `unlearner.run(model=..., datasets=..., attack_result=..., detection_result=...)`
- shared training path is `train_torch_model(...)`
- `datasets` contains:
  - `train`
  - `val`
  - `test`
  - `test_triggered`
  - `test_clean_labels`
- oracle poison labels are in `attack_result.poison_indices`
- detector output may expose:
  - `suspect_indices`
  - `sample_flags`
  - `sample_ranking`
  - `sample_scores`
- IDS default path is numeric-only

Do not invent a new training stack.

## Track Meaning

- `noop`: keep attacked model unchanged
- `oracle_retrain`: remove true poisoned samples, train a fresh model from scratch
- `detected_retrain`: remove detector-selected samples, train a fresh model from scratch

Use:
- `noop` as lower bound
- `oracle_retrain` as practical upper bound for `detect -> remove -> retrain`
- `detected_retrain` as the first real baseline

## Files To Build

```text
src/unlearning/
  BASE_UNLEARNING_BLUEPRINT.md
  __init__.py
  types.py
  base.py
  utils.py
  retrain.py
```

## Required Types

### `ForgetSet`

Fields:
- `indices`
- `scores`
- `flags`
- `source`
- `notes`

Allowed `source`:
- `none`
- `oracle_poison`
- `detector_flags`
- `detector_topk`
- `manual`

### `UnlearningContext`

Fields:
- `model`
- `datasets`
- `attack_result`
- `detection_result`
- `model_cfg`
- `train_cfg`
- `seed`
- `device`
- `num_classes`
- `class_names`
- `run_dir`
- `method_cfg`

Compatibility rule:
- public `run(...)` may accept the raw kwargs currently passed by `run.py`
- internally convert them to `UnlearningContext`

### `UnlearningArtifacts`

Fields:
- `summary_json`
- `metrics_before_json`
- `metrics_after_json`
- `forget_indices_npy`
- `retain_indices_npy`
- `checkpoint_dir`
- `extra_files`

### `UnlearningResult`

Fields:
- `method_name`
- `track_type`
- `status`
- `seed`
- `runtime_sec`
- `forget_set_source`
- `num_removed`
- `num_retained`
- `metrics_before`
- `metrics_after`
- `summary_metrics`
- `removed_indices`
- `retain_indices`
- `artifacts`
- `deviation_note`

## Required Metrics

### Before

- `clean/test/accuracy_before`
- `backdoor/asr_before`
- `backdoor/accuracy_before`

### Forget-set quality

Only when oracle poison indices exist:
- `unlearning/forget_precision`
- `unlearning/forget_recall`
- `unlearning/forget_f1`
- `unlearning/forget_size`
- `unlearning/retain_size`
- `unlearning/remove_fraction`

### After

- `clean/test/accuracy_after`
- `backdoor/asr_after`
- `backdoor/accuracy_after`
- `clean/val_accuracy_best_after`
- `unlearning/runtime_sec`

### Delta

- `unlearning/delta_clean_accuracy`
- `unlearning/delta_asr`
- `unlearning/delta_backdoor_accuracy`

Definitions:
- `delta_clean_accuracy = accuracy_after - accuracy_before`
- `delta_asr = asr_after - asr_before`

Good unlearning:
- `delta_asr << 0`
- `delta_clean_accuracy` close to `0`

## Required Artifacts

Save at least:
- `summary.json`
- `metrics_before.json`
- `metrics_after.json`
- `forget_indices.npy`
- `retain_indices.npy`
- checkpoint directory for the unlearned model

Suggested root:

```text
artifacts/unlearning/<method>_<timestamp>/
```

## Base Class Contract

Create `BaseUnlearner` with:
- `__init__(cfg)`
- `run(...)`
- `_build_context_from_kwargs(...)`
- `_validate_context(context)`
- `_resolve_forget_set(context)`
- `_run_impl(context, forget_set)`
- `_evaluate_model(model, datasets, target_label, prefix)`
- `_save_artifacts(result, context)`

Rules:
- `run(...)` is the only public entrypoint
- seed, timing, validation, artifact saving live in `run(...)`
- method logic lives in `_run_impl(...)`

## Forget-Set Rules

### `noop`

Return:
- empty `indices`
- `source="none"`

### `oracle_retrain`

Use:
- `attack_result.poison_indices`

Fail if missing.

### `detected_retrain`

Preferred order:
1. `detection_result.suspect_indices`
2. `sample_flags`
3. `sample_ranking[:k]` when config requests top-k

Fail if:
- detector-based source requested
- `detection_result` missing
- detector produced no usable suspect output

Do not silently fallback to oracle.

## Retrain Contract

Create `RetrainFromScratchUnlearner`.

Behavior:
1. evaluate attacked model before unlearning
2. resolve forget set
3. split attacked train into forget / retain
4. create a fresh model with the same architecture config
5. train from scratch on retain split using `train_torch_model(...)`
6. evaluate after unlearning
7. save artifacts

Hard rules:
- fresh init only, no finetune
- same model family
- same base train config unless explicit override
- same `val`, `test`, `test_triggered`, `test_clean_labels`
- same attack target label for ASR

Need config fields like:
- `name`
- `track_type`
- `forget_set.source`
- `forget_set.topk`
- `allow_empty_forget_set`
- `min_retained_samples`
- `train_overrides`
- `artifact_root`

## `noop` Contract

Create `NoOpUnlearner`.

Behavior:
1. evaluate current attacked model
2. copy before metrics into after metrics
3. return `status="skipped"`

`noop` must still return a valid `UnlearningResult`.

## Evaluation Rules

Reuse existing metric semantics:
- clean accuracy on `test`
- ASR on `test_triggered` against `test_clean_labels`

Do not invent a second ASR definition inside unlearning.

Prefer factoring shared evaluation code out of `src/models/train.py` if needed.

## Failure Rules

Fail loudly when:
- requested forget-set source is unavailable
- all samples are removed
- retained set is below configured minimum
- `train`, `val`, or `test` is missing

Optional explicit config:
- `allow_empty_forget_set=true`

If empty forget set is allowed:
- retrain may run on the full attacked train split
- log `num_removed=0`

## Registry Rule

Eventually expose:

```python
UNLEARNING_REGISTRY = {
    "none": NoOpUnlearner,
    "noop": NoOpUnlearner,
    "retrain": RetrainFromScratchUnlearner,
    "oracle_retrain": RetrainFromScratchUnlearner,
    "detected_retrain": RetrainFromScratchUnlearner,
}
```

Do not keep `baeraser` as a fake alias unless it becomes a real method.

## Implementation Order

### Stage 1

Implement:
- `types.py`
- `base.py`
- `utils.py`
- `NoOpUnlearner`

Done when:
- `noop` returns a full `UnlearningResult`
- before/after metrics exist

### Stage 2

Implement:
- `RetrainFromScratchUnlearner`
- config variants for:
  - `oracle_retrain`
  - `detected_retrain`

Done when:
- retain split is correct
- fresh model retrains correctly
- before/after/delta metrics are saved

### Stage 3

Integrate unlearning output into benchmark summary.

Do this only after Stage 2 works.

## Smoke Test

First cheap path:
- `model=mlp`
- `attack=badnets`
- `detection=spectral_signatures` or `none`
- `unlearning=oracle_retrain`
- 1 seed
- low epoch count

Check:
- `forget_indices.npy` exists
- `retain_indices.npy` exists
- checkpoint exists
- `clean/test/accuracy_after` exists
- `backdoor/asr_after` exists

Then test:
- `unlearning=detected_retrain`

## Non-Goals

Do not do in v1:
- paper methods
- finetune-after-removal
- pruning-based repair
- plugin abstractions
- categorical-specialized path
- callback-heavy design

## Agent Notes

- keep code flat
- prefer local helpers over deep abstractions
- mirror `src/detection` style where useful
- optimize for correct metrics and contracts, not elegance
- if later paper/code conflict appears, follow official code and log deviation

## One-Line Summary

Rebuild `src/unlearning` around one shared contract for `forget set -> retrain/evaluate -> artifacts`, and only support `noop`, `oracle_retrain`, and `detected_retrain` before anything more ambitious.
