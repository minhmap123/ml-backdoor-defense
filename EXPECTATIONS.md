# Project Expectations

This project is being executed as conference-style empirical research, not as production software.

## 1) Evidence Standard
- Results must be reproducible enough for paper tables, ablations, and rebuttal follow-up.
- Claims about a baseline should be backed by:
  - paper link,
  - official code link when available,
  - exact local config,
  - seed,
  - core metrics,
  - recorded deviations.

## 2) Implementation Standard
- Preferred pattern for model baselines:
  - official or upstream-derived architecture core,
  - thin local wrapper that matches this repo's interfaces,
  - local training/evaluation pipeline for apples-to-apples comparison.
- Do not default to full paper-only rewrites when official code exists.
- Do not import entire upstream experiment frameworks unless required to validate a result.

## 3) Fairness Standard
- Keep preprocessing, train/val/test protocol, and metrics comparable across baselines.
- Do not give one model a materially different optimization or evaluation pipeline unless the paper requires it.
- If a model requires a special setting, record it explicitly and treat it as a controlled deviation.

## 4) Deviation Standard
- Every deviation from paper or official code must state:
  - what changed,
  - why it changed,
  - expected impact on metrics or runtime,
  - where the evidence is stored.

## 5) Minimum Reproducibility Artifacts
- config file or resolved config dump
- seed
- saved checkpoint or model state
- metric summary
- short session note or experiment note

## 6) Practical Rule
- When there is a tradeoff between elegant code and faithful reproduction, prefer faithful reproduction for baselines used in the paper.
