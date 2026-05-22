# AGENTS.md

Project-level rules for `ml-backdoor-defense` — a backdoor-detection benchmark on tabular IDS models.
This file overrides home-level defaults inside this repository.

## 1. What this project is

- Benchmark of **model-level backdoor detectors** on tabular intrusion-detection datasets.
- Pipeline: clean dataset → poisoned dataset (attack) → trained victim model → detector run → metrics.
- Two stages in [run.py](run.py), selected by `pipeline.stage`:
  1. `attack_train` — build poisoned data, train victim, dump artifacts.
  2. `detection` — load the trained victim, run a detector, write decision metrics.
- Datasets in scope: `iotid20`, `cic_ids2017`, `cic_ids2018`, `cic_iot_2023` (numeric-only baseline, see [src/data/DATA_FORMAT_SPEC.md](src/data/DATA_FORMAT_SPEC.md)).
- Models in scope: MLP, Tabular ResNet, TabNet, FT-Transformer, SAINT (see [src/models/MODEL_BASE_SPEC.md](src/models/MODEL_BASE_SPEC.md)).
- Attacks in scope: BadNets, TabDoor, CatBack (see [src/attacks/](src/attacks/)).
- Detectors in scope: NC, MM-BD, MLBD, PT-RED and their `*_cso` variants (see [src/detection/BASE_DETECTOR_BLUEPRINT.md](src/detection/BASE_DETECTOR_BLUEPRINT.md)).
- **Out of scope** (do not re-introduce without an explicit ask): machine unlearning, spectral signatures, sample-localization metrics, UNICORN, BTIDBF.

## 2. Required reading on first touch

Load in this order, then start work:
1. This file.
2. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — methods catalog, paper/official-code links, reproduction protocol.
3. Spec files for the layer being touched:
   - data → [src/data/DATA_FORMAT_SPEC.md](src/data/DATA_FORMAT_SPEC.md)
   - models → [src/models/MODEL_BASE_SPEC.md](src/models/MODEL_BASE_SPEC.md)
   - detection → [src/detection/BASE_DETECTOR_BLUEPRINT.md](src/detection/BASE_DETECTOR_BLUEPRINT.md)

If `analysis/*.md` has a relevant report (e.g. `MLBD_vs_MMBD_analysis.md`, `NC_NCCSO_analysis.md`), read it before re-running the same comparison.

## 3. Project mode

- Mode: `research-prototype` with `reproduction-first` priority.
- Target standard: **conference-style empirical research** — reproducible tables, explicit deviations, fair cross-method comparison.
- Favor changes that unblock experiments over production hardening.

## 4. Priority order

When goals conflict, resolve in this order:

1. **Experimental correctness** of attack injection, victim training, and detector decision logic.
2. **Fairness across baselines** — same preprocessing, same train/val/test protocol, same metric definitions.
3. **Reproducibility** — seed, config, artifact layout, deviation notes.
4. **Iteration speed** for ablations.
5. **Code quality / maintainability.**
6. Production hardening, input validation pipelines, formal security frameworks — only if explicitly requested.

## 5. Reproduction protocol (mandatory)

Order of truth when reproducing a method:

1. Paper methodology and reported setup.
2. Official author code (links in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) §4).
3. Strong community reimplementation.
4. Local adaptation in this repo.

Rules:

- If paper and official code disagree, follow the official code and **note the mismatch** in the commit message or an `analysis/*.md` note.
- Keep original hyperparameters, preprocessing, and evaluation protocol unless a change is required to run here.
- Every deviation must record: **what changed / why / expected metric or runtime impact / where it is logged**.
- For baselines used in the paper: prefer faithful reproduction over stylistic code purity. Use upstream architecture core + thin local wrapper to match repo contracts, not a full paper-only rewrite.
- Do not import an entire upstream training/experiment framework unless it is required to validate a result — port the architecture, keep this repo's training/eval pipeline.

## 6. Experimental contract (do not silently break)

These are the load-bearing invariants of the benchmark — changing them invalidates prior runs:

- **Pipeline shape:** the two stages, their artifact directories, and `summary.json` schema in [run.py](run.py).
- **Fair-comparison constants:** seed, train/val/test split logic, `class_weight_mode`, `selection_metric`, target label per dataset (see [scripts/run_full_benchmark.py](scripts/run_full_benchmark.py)).
- **Detector contract:** the fields on `DetectorResult` (infected flag, target/source class, decision score, threshold, margin, threshold direction). New detectors must populate all of them.
- **Model contract:** `forward / forward_features / forward_logits` and required metadata in `MODEL_BASE_SPEC.md`.
- **Metric set:** original test accuracy, attack clean accuracy, ASR, infected prediction, predicted target class, target-class score, decision score/threshold/margin, runtime.

If a change touches any of the above, say so explicitly and propose a migration for prior runs (re-run, rescore, or drop).

## 7. How to run things

Single config:

```bash
python run.py data=iotid20 model=mlp attack=badnets detection=neural_cleanse
```

Full benchmark for one dataset (attack × model × detector grid, with `skip_existing`):

```bash
python scripts/run_full_benchmark.py --dataset iotid20
```

Summarize the most recent benchmark run for any dataset with the same artifact layout:

```bash
python scripts/summarize_benchmark.py
```

Artifacts land under `results/benchmark_runs/<dataset>/<timestamp>/artifacts/{attack_train,detection}/...` and reports under `.../reports/`.

## 8. Coding rules for this repo

- Keep implementations minimal and easy to modify for ablations.
- Do not over-engineer APIs, validation layers, abstractions, or framework scaffolding.
- Prefer direct, local fixes over broad refactors.
- Default to simple `assert`s for sanity checks; do not build strict validation pipelines.
- Do not add features, comments, or backwards-compatibility shims that the current task does not need.
- Never re-introduce out-of-scope methods (§1) without an explicit instruction.

## 9. Script style for `scripts/` and `analysis/`

(Applies to experiment / analysis / one-shot scripts, **not** library code under `src/`.)

- Hardcode paths and hyperparameters at the top of the file. No `argparse` unless reuse across many contexts is obvious.
- No type hints on function signatures unless they add real clarity.
- No wrapper functions that only wrap one or two lines — inline them.
- Flat, sequential flow over layered abstractions (collect → score → report as plain loops, not classes or pipelines).
- No section-header comments (`# ── Data loading ──`) unless the file exceeds ~150 lines.
- Omit `from __future__ import annotations`, `typing`, `Any`, `math` unless actually needed.
- A reader should be able to follow the whole script top-to-bottom without jumping between functions.

## 10. Validation policy

- Default: run the smallest relevant check for the changed code path; verify a key metric still computes.
- One-shot smoke before launching long runs: a single `(data, model, attack, detector)` config end-to-end.
- Full test suites, exhaustive edge-case validation, enterprise reliability gates: only if requested.
- If tests are slow or absent, leave a short manual verification note in the response.

## 11. Long-running experiments

- The full benchmark grid takes hours. Before suggesting a re-run, check whether `skip_existing=true` already covers what is needed.
- When a benchmark is running, prefer offline work that does not contend for the GPU: analysis of finished runs, doc edits, summarizer/reporting tweaks, new detector implementation scaffolding without training.
- Never delete a `results/benchmark_runs/<...>/` directory without explicit confirmation.

## 12. Reporting and notes

Persistent notes go in:

- `analysis/*.md` — comparative analyses, failure investigations, progress reports.
- `results/benchmark_runs/<run>/reports/` — auto-generated per-run reports (`summarize_benchmark.py`).
- Commit message — short description plus any deviation from paper / official code.

A reproduced-baseline claim is not ready until it has: paper link, official-code link, local config (or resolved config dump), seed, saved checkpoint / model state, core metric summary, and recorded deviations.

## 13. Security posture

- Treat this repo as research code, not production.
- Baseline hygiene only: no hardcoded secrets, no unsafe system operations outside task scope, no publishing sensitive artifacts.
- Do not introduce formal security frameworks or auth/validation layers by default.

## 14. Communication style

- Concise, action-oriented; suggest the smaller experiment when it answers the question.
- Call out explicitly when a request shifts from research prototype to production requirements, or when it would break an invariant in §6.
- When a paper-vs-code disagreement is hit, surface it instead of silently picking one side.
