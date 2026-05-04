# IoTID20 Staged Benchmark

This benchmark is staged so conference-facing experiments can reuse artifacts
without retraining or accidentally joining results from another run.

## Stage Contract

Each stage writes `summary.json`, `status.txt`, `resolved_config.yaml`,
`runtime_sec`, and `seed` under an explicit run-specific directory.

```text
results/benchmark_runs/<stamp>/artifacts/
  attack_train/<mode>__<attack>__<seed>__<model>/
    checkpoint/
    attack_result/
    data/datasets.npz
    data/*_sample_provenance.csv
    metadata.json
    model_cfg.json
    train_cfg.json
    model_metrics.json
    summary.json
  detection/<attack_train_run_id>__det-<detector>/
    summary.json
    detector_summary.json
    detection/<detector>/*
  unlearning/<detection_run_id>__unlearn-<method>/
    summary.json
    unlearner_summary.json
    unlearning/<method>/*
```

## Execution Flow

1. `attack_train` loads raw IoTID20, prepares split, balances train data,
   injects the configured backdoor, trains the model, then saves checkpoint,
   dataset arrays, attack result, scaler metadata, and sample provenance.
2. `detection` loads only the `attack_train` artifact and checkpoint, then
   writes detector output under its own artifact directory.
3. `unlearning` loads the same `attack_train` artifact plus one detection
   artifact, then runs the selected unlearning method.

The benchmark script never uses "latest directory wins". Every summary row
points to the exact attack, detection, and unlearning artifact directories.

## Seed Semantics

The single top-level `seed` is propagated to:

- IoTID20 train/val/test split.
- Train balancing and final permutation.
- Poison sampling through `attack.seed`.
- Model initialization and dataloader shuffle through `train.seed`.
- Detection through `detection.seed`.
- Unlearning through `unlearning.seed`.

This means a run key `(mode, attack, seed, model)` defines exactly one attacked
model artifact that can be reused by all detector and unlearning stages.

## Detector/Unlearning Policy

Detector outputs are treated by capability, not by name. Detector-guided
unlearning runs only when the detection artifact actually contains
`suspect_indices.npy` or sample-score flags. Class-level detectors such as
Neural Cleanse, MM-BD, MLBD, and CSO variants remain class-level evidence and
are reported with `unlearning=none` unless a separate bridge method is added.

`clean` controls are intentionally deferred. The current benchmark script only
runs `attacked` mode.
