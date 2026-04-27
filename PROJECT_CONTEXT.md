# PROJECT_CONTEXT

## 1) Project Snapshot
- Project: Backdoor patching using machine unlearning
- Domain: AI Security benchmark (tabular + IoT intrusion datasets)
- Main objective: evaluate how effectively machine unlearning can remove backdoor behavior while preserving clean performance.
- Research style: `research-prototype` with `reproduction-first` priority.
- Target execution standard: conference-style empirical research with explicit reproducibility and deviation tracking.

## 1.1) Implementation Stance
- Default stance for conference-facing experiments:
  - use paper methodology first,
  - use official author code as the architecture source of truth when available,
  - adapt with thin local wrappers instead of rewriting whole models from scratch.
- Preferred model integration pattern:
  - upstream or official core implementation for architecture details,
  - local wrapper for repo contracts (`forward`, `forward_features`, `forward_logits`, metadata, save/load),
  - local training/evaluation pipeline for fair comparison across attacks, detection, and unlearning.
- Avoid for baseline models unless there is a clear blocker:
  - full reimplementation from paper when official code exists,
  - importing an entire upstream training stack into this repo,
  - changing architectural details for convenience without logging the change.
- Any change from paper or official implementation must be logged with:
  - what changed,
  - why it changed,
  - expected metric impact,
  - where it is recorded (session notes, experiment notes, or commit message).

## 2) Scope
- Datasets in scope:
  - IoTID20
  - CIC-IDS2017
  - CSE-CIC-IDS2018
  - CIC-IoT-2023
  
- Candidate model families in scope:
  - MLP
  - Tabular ResNet
  - TabNet
  - FT-Transformer
  - SAINT

## 3) Evaluation Metrics (primary)
- Original test accuracy
- Attack clean accuracy
- Attack success rate (ASR)
- Detection precision / recall / F1 when poisoned indices are available
- Suspect ranking quality (top-k recall) for localization experiments
- Unlearned clean accuracy
- Unlearned attack success rate
- Runtime / unlearning time

Notes:
- Log metrics to W&B when available.
- Keep per-run `seed`, `config`, and method tags for reproducibility.

## 4) Methods Catalog

### 4.1 Backdoor Attacks
- BadNets
  - Paper: https://arxiv.org/abs/1708.06733

- TabDoor (tabular transformer backdoor)
  - Paper: https://arxiv.org/abs/2311.07550
  - Official source code: https://github.com/bartpleiter/tabular-backdoors

- CatBack (categorical-encoding-based tabular backdoor)
  - Paper: https://arxiv.org/abs/2511.06072
  - NDSS page: https://www.ndss-symposium.org/ndss-paper/catback-universal-backdoor-attacks-on-tabular-data-via-categorical-encoding/
  - Official source code: https://github.com/catback-tabular/catback.git

### 4.2 Backdoor Detection 
- Spectral Signatures
  - Paper: https://papers.nips.cc/paper_files/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf
  - Official source code: https://github.com/MadryLab/backdoor_data_poisoning

- Neural Cleanse (NC)
  - Role: reverse-engineering detector using a small trigger-mask norm as anomaly evidence.
  - Paper: https://doi.org/10.1109/SP.2019.00031
  - Official code: https://github.com/bolunwang/backdoor

- UNICORN
  - Full title: UNICORN: A Unified Backdoor Trigger Inversion Framework
  - Paper: https://openreview.net/forum?id=Mj7K4lglGyj
  - Official code: https://github.com/RU-System-Software-and-Security/UNICORN

- BTIDBF
  - Full title: Towards reliable and efficient backdoor trigger inversion via decoupling benign features
  - Paper: https://openreview.net/forum?id=Tw9wemV6cb
  - Official code: https://github.com/xuxiong0214/BTIDBF

- MM-BD:
  - Paper: https://xplorestaging.ieee.org/document/10646729/
  - Official code: https://github.com/wanghangpsu/MM-BD

- MLBD
  - Role: maximum-logit backdoor detector used as a baseline in the CSO paper.

- CSO
  - Paper: https://arxiv.org/pdf/2512.08129
  
- NC-CSO
  - Role: CSO-augmented variant of Neural Cleanse.
- MMBD-CSO
  - Role: CSO-augmented variant of MM-BD.
- MLBD-CSO
  - Role: CSO-augmented variant of MLBD.

### 4.3 Machine Unlearning / Backdoor Removal
- Bad Teaching (incompetent teacher)
  - Paper: https://arxiv.org/abs/2205.08096
  - Code: https://github.com/vikram2000b/bad-teaching-unlearning
- Anti-Backdoor Learning: Training Clean Models on Poisoned Data
  - Paper: https://arxiv.org/abs/2110.11571
  - Official code: https://github.com/bboylyg/ABL
- Reconstructive Neuron Pruning for Backdoor Defense
  - Paper: https://arxiv.org/abs/2305.14876
  - Official code: https://github.com/bboylyg/RNP



### 4.4 Candidate Model Families
- MLP
  - Role: baseline feed-forward network for tabular intrusion features.
- Tabular ResNet
  - Reference: RTDL ResNet from `Revisiting Deep Learning Models for Tabular Data`
  - Paper: https://arxiv.org/abs/2106.11959
  - Official code: https://github.com/yandex-research/rtdl-revisiting-models
- TabNet
  - Paper: https://arxiv.org/abs/1908.07442
  - Code: https://github.com/dreamquark-ai/tabnet
- FT-Transformer
  - Paper: https://arxiv.org/abs/2106.11959
  - Official code: https://github.com/yandex-research/rtdl-revisiting-models
  - Expected local integration: RTDL-derived core + repo wrapper, not a paper-only rewrite.
- SAINT
  - Paper: https://arxiv.org/abs/2106.01342
  - Official source code: https://github.com/somepago/saint
  - Expected local integration: official-source-guided port + repo wrapper.

## 5) Reproduction Protocol (must follow)
- Priority order for implementation fidelity:
  1. Paper methodology
  2. Official author code (if public)
  3. Strong community reimplementation
  4. Local adaptation
- If paper/code conflicts, prefer official code and record mismatch.
- Every deviation must include:
  - what changed
  - why changed
  - expected metric impact
- Conference-facing baseline policy:
  - architecture details should follow official code whenever available,
  - local code should wrap or minimally port upstream logic instead of inventing a fresh implementation,
  - train/eval protocol should remain unified inside this repo so all baselines share the same attack/detection/unlearning pipeline,
  - fairness matters more than stylistic code purity.
- Minimum documentation expected before claiming a reproduced baseline is ready:
  - source links for paper and official code,
  - local config used,
  - dataset preprocessing assumptions,
  - known deviations,
  - minimal smoke verification result.

## 6) Quick Resume Checklist
On a fresh session:
1. Read `AGENTS.md` then this file.
2. Read `EXPECTATIONS.md` to align implementation and reporting expectations.
3. Open `session_memory/SESSION_LOG.md` (latest entry first).
4. Open `session_memory/DECISIONS.md` and unresolved items.
5. Pick the top pending item in `session_memory/NEXT_STEPS.md`.
6. Run a minimal smoke command before long experiments.
