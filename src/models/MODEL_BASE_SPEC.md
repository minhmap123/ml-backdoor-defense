# src/models Base Tabular Model Spec

## Goal
Define one compact contract for all tabular models in this repo.

Required properties:
- multi-class by default,
- logits out, no softmax in model,
- supports attack / detection / unlearning,
- supports MLP, Tabular ResNet, TabNet, FT-Transformer, SAINT,
- small enough for agent consumption.

## What the Base Must Support

### Inputs
Accept exactly these input forms:
- `x`
- `{"x": ...}`
- `{"x_num": ..., "x_cat": ...}`
- `(x, y)` from dataloaders, where `y` is used by the training / metrics code outside the model

For mixed tabular models, keep `x_num` and `x_cat` explicit until the subclass packs them.

### Outputs
- `forward(x) -> logits` with shape `[batch_size, num_classes]`
- `forward_features(x) -> features`
- `forward_logits(features) -> logits`

Default shape is multi-class. If a subclass internally uses a different shape, normalize before returning.

### Metadata
Every model should expose:
- `name`
- `model_family`
- `d_in`
- `d_out` / `num_classes`
- `hidden_dim` if applicable
- `num_parameters`
- optional structured-input fields: `num_numeric_features`, `num_categorical_features`, `categorical_cardinalities`, `embedding_dim`

### Constructor / API
Recommended shape:
- `__init__(*, d_in, d_out, **model_kwargs)`
- `d_out` always means number of classes
- model-specific kwargs stay in the subclass or config

Recommended public methods:
- `forward`
- `forward_features`
- `forward_logits`
- `get_model_metadata`
- `reset_parameters` if supported

## Feature Contract
`forward_features()` should return the last useful hidden state before the classifier head.

Family guidance:
- MLP: last hidden block output
- Tabular ResNet: penultimate/residual trunk output
- TabNet: decision representation
- FT-Transformer: transformer output before final head
- SAINT: contextualized token embedding / pooled sequence output

Detectors like Spectral Signatures need this path directly.

## Save / Load Contract
Use run folders, explicit files, and lightweight metadata.

Recommended contents:
- `model_state.pt` or `model_state_dict.pt`
- `config.json`
- `metadata.json`
- optional `optimizer.pt`
- optional `metrics.json`

Recommended API:
- `save_model(model, output_dir, config=None, metadata=None, optimizer=None, metrics=None)`
- `load_model(model_class, checkpoint_dir, strict=True)`

Loading rule:
- reconstruct the model from saved config first,
- then load state dict,
- do not rely on implicit defaults.

## Metrics Contract
Always evaluate both clean and backdoor behavior.

### Required metrics
- `clean_accuracy`
- `backdoor_accuracy`
- `attack_success_rate` (ASR)
- `train_accuracy`
- `val_accuracy`
- `clean/classification_report`

### Definitions
- `clean_accuracy = correct_predictions_on_clean_test / clean_test_samples`
- `attack_success_rate = target_predictions_on_triggered_non_target_samples / triggered_non_target_samples`
- `backdoor_accuracy` is accuracy on the triggered set; if ambiguous, prefer ASR as the primary backdoor metric

### Clean evaluation
For multiclass runs, the clean view should also log a classification report with:
- precision
- recall
- F1-score
- support
- macro avg
- weighted avg

### Backdoor evaluation
For the triggered view:
- take non-target test samples,
- apply the trigger,
- measure ASR against the attacker target class,
- optionally log `backdoor/accuracy` if you build a full triggered test set.

### Detection metrics
If poisoned indices are known, also log:
- `detection/precision`
- `detection/recall`
- `detection/f1`
- optional `top-k recall`

## Utilities
`src/models/utils.py` should stay architecture-agnostic.

Suggested helpers:
- `split_to_numpy(split)`
- `set_seed(seed)`
- `resolve_device(device)`
- `save_model(...)`
- `load_model(...)`
- `normalize_logits(logits)`

## Training Contract
`src/models/train.py` should:
- build a model from config,
- move it to device,
- train with classification loss,
- save checkpoints,
- log train/val/test metrics,
- use `utils.py` for seed/device/checkpoint helpers.

## Implementation Order
1. `base.py`
2. `utils.py`
3. `__init__.py`
4. `train.py`
5. `mlp.py`
6. `tabular_resnet.py`
7. `tabnet.py`
8. `ft_transformer.py`
9. `saint.py`

## Base Is Ready When
- a dummy subclass returns multi-class logits,
- tensor / dict / tuple inputs work,
- feature extraction exists,
- metadata is queryable,
- save/load uses explicit checkpoint folders,
- clean/backdoor metrics are defined clearly.
