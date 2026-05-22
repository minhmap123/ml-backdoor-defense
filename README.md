# Backdoor Detection on Tabular IDS Models

Benchmark of **model-level backdoor detectors** on tabular intrusion-detection datasets.
Every experiment follows a two-stage pipeline: (1) poison a dataset and train a victim model, (2) run a detector on the saved model and report the infection decision with supporting scores.

---

## Scope

### Datasets

| Dataset | Classes | Samples (approx.) |
|---|---|---|
| IoTID20 | 5 | 625 K |
| CSE-CIC-IDS2018 | 8 | 16 M |
| CIC-IDS2017 | 15 | 2.8 M |
| CIC-IoT-2023 | 8 | 7.2 M |

### Attacks

| Attack | Type | Poison rate |
|---|---|---|
| **BadNets** | Fixed-value feature trigger | 5% |
| **TabDoor** | Learned continuous trigger | 2% |
| **CatBack** | Categorical-encoding backdoor | 2% |

### Victim models

MLP · Tabular ResNet · TabNet · FT-Transformer · SAINT

### Detectors

| Detector | Family | CSO variant |
|---|---|---|
| Neural Cleanse (NC) | Trigger inversion | NC-CSO |
| MM-BD | Modal-mass trigger inversion | MMBD-CSO |
| MLBD | Maximum-logit baseline | MLBD-CSO |
| PT-RED | Perturbation-based | PT-RED-CSO |

CSO variants apply the **Class Score Ordering** post-processing from [arXiv 2512.08129](https://arxiv.org/pdf/2512.08129) on top of each base detector.

---

## How to run

**Single experiment (attack train + detection):**

```bash
# Stage 1 — poison and train
python run.py data=iotid20 model=mlp attack=badnets detection=none \
    pipeline.stage=attack_train pipeline.run_id=my_run

# Stage 2 — detect
python run.py data=iotid20 model=mlp attack=badnets detection=pt_red \
    pipeline.stage=detection pipeline.run_id=my_run \
    pipeline.attack_train_artifact_dir=artifacts/pipeline/my_run/attack_train
```

**Full benchmark grid (all attacks × models × detectors for one dataset):**

```bash
python scripts/run_full_benchmark.py --dataset iotid20
# Available: iotid20 | cic_ids2018 | cic_ids2017 | cic_iot_2023
```

**Summarize a completed run:**

```bash
python scripts/summarize_benchmark.py
```

Reports land under `results/benchmark_runs/<dataset>/<timestamp>/reports/`.

---

## Experimental results

All numbers are averaged over **5 victim models** (MLP, ResNet, TabNet, FT-Transformer, SAINT) trained with seed 7.
Metrics: clean test accuracy and macro-F1 on the unmodified test set; ASR = attack success rate on triggered test samples.

### Victim model performance

| Dataset | Attack | Clean Acc | Macro F1 | ASR |
|---|---|---|---|---|
| IoTID20 | BadNets | 81.5% | 77.0% | **100.0%** |
| IoTID20 | CatBack | 81.3% | 76.9% | **100.0%** |
| IoTID20 | TabDoor | 81.6% | 77.1% | 95.0% |
| CIC-IDS2018 | BadNets | 99.6% | 93.2% | **100.0%** |
| CIC-IDS2018 | CatBack | 99.6% | 93.8% | **100.0%** |
| CIC-IDS2018 | TabDoor | 99.5% | 93.5% | **100.0%** |

Clean accuracy and F1 are stable across all attacks (backdoor training does not degrade model quality).

### Detector performance

Each cell = **detection rate** / **target-class ID rate** (n = 15 runs: 5 models × 3 attacks).
Detection rate: fraction of backdoored models correctly flagged as infected.
Target-class ID rate: fraction of runs where the predicted target class matches the true one.

| Detector | IoTID20 | CIC-IDS2018 |
|---|---|---|
| **MMBD-CSO** | 93% / 33% | **100% / 100%** |
| **PT-RED** | 87% / 27% | **100% / 100%** |
| **PT-RED-CSO** | 73% / 20% | **100% / 100%** |
| **MLBD-CSO** | **100% / 33%** | 80% / 100% |
| MM-BD | 67% / 20% | 80% / 93% |
| MLBD | 87% / 13% | 47% / 80% |
| Neural Cleanse | 20% / 27% | 33% / 100% |
| NC-CSO | 13% / 27% | 20% / 100% |

**Key observations:**

- MMBD-CSO and PT-RED are consistently strong across both datasets, reaching 100% detection on CIC-IDS2018.
- MLBD-CSO achieves perfect detection on IoTID20 but drops on CIC-IDS2018, suggesting sensitivity to dataset characteristics.
- NC and NC-CSO underperform on tabular data; trigger-inversion via optimization is ineffective when the feature space lacks spatial structure.
- CSO augmentation consistently improves or matches the base detector for MLBD and MM-BD; the benefit for PT-RED reverses on IoTID20, warranting further investigation.
- Target-class identification is harder on IoTID20 (low F1 dataset) than on CIC-IDS2018 (high F1 dataset), decoupled from the infected/clean decision.

### Post-hoc analysis: NC decision rule (ratio test)

NC and NC-CSO flag a class as backdoored when its reversed-trigger mask norm is an outlier among all K class norms.
The original rule uses a MAD-based threshold; a simpler **ratio test** (`min_norm / median_norm < θ`) was evaluated post-hoc from saved `optimization_trace.json` artifacts without re-running optimization (n = 68 runs across IoTID20 and CIC-IDS2018).

| Detector | Dataset | MAD > 2.0 | ratio < 0.3 | ratio < 0.5 |
|---|---|---|---|---|
| NC | IoTID20 | 7% | 13% | 27% |
| NC-CSO | IoTID20 | 0% | 0% | 27% |
| NC | CIC-IDS2018 | 33% | **100%** | 100% |
| NC-CSO | CIC-IDS2018 | 20% | **100%** | 100% |

Numbers report the fraction of runs where the detector both detects infection *and* identifies the correct target class.

**Interpretation:**

- On **CIC-IDS2018**, NC's low detection rate (20–33%) is entirely a threshold problem: `min/median` ratios are consistently well below 0.3, so the trigger class is clearly separable in norm space. Switching to `ratio < 0.3` recovers **100%** accuracy for both NC and NC-CSO (Δ = +10 and +12 respectively).
- On **IoTID20**, `min/median` ratios cluster between 0.3–0.7 across all runs — no outlier structure. The bottleneck is the optimization itself failing to recover a meaningful trigger in the 5-class, lower-F1 setting; no threshold adjustment compensates for a flat norm distribution.

> **In progress:** CIC-IDS2017 and CIC-IoT-2023 benchmarks are running; results will be added upon completion.

---

## Artifact layout

```
results/benchmark_runs/<dataset>/<timestamp>/
  artifacts/
    attack_train/<run_id>/    # poisoned datasets, model checkpoint, attack result
    detection/<run_id>/       # detector decision, scores, resolved config
  reports/
    training_summary_by_attack_model.csv
    detection_summary_by_detector_attack.csv
    detection_summary_by_detector.csv
  logs/
```

Each stage writes a `summary.json` with status, seed, config path, and all reported metrics.
W&B logging is enabled by default; set `wandb.enabled=false` to disable.
