# AGENTS.md

Project-level agent rules for:
`/home/minu2k5/projects/Backdoor patching using machine unlearning`

This file overrides broader/home-level defaults for this repository.

## Startup context loading (mandatory)
- At the start of each new session, load context in this order:
  1. `AGENTS.md` (this file)
  2. `PROJECT_CONTEXT.md`
  3. `session_memory/SESSION_LOG.md` (latest entry first)
  4. `session_memory/DECISIONS.md` (focus on unresolved items)
  5. `session_memory/NEXT_STEPS.md`
- If time is limited, read `PROJECT_CONTEXT.md` and `session_memory/NEXT_STEPS.md` first.
- Treat these files as the canonical local memory source for this repository.

## Project mode
- Operate in `research-prototype` mode.
- Primary goal: fast experiment iteration and idea validation.
- Favor changes that unblock experiments over production hardening.

## Priority order
1. Experimental correctness of core logic (attack/unlearning behavior)
2. Speed of iteration and reproducibility
3. Code quality and maintainability
4. Strict input/output hardening and security framework compliance

## Coding rules for this repo
- Keep implementations minimal and easy to modify for ablations.
- Do not over-engineer APIs, validation layers, or architecture.
- Avoid adding heavy security/framework scaffolding unless explicitly requested.
- Default to simple assertions and lightweight sanity checks instead of strict validation pipelines.
- Prefer direct, local fixes over broad refactors.

## Reproduction-first rules
- When reproducing a paper, prioritize fidelity over creativity.
- Follow this order of truth:
  1. Paper methodology and reported setup
  2. Official public implementation from paper authors (if available)
  3. Widely used community reimplementations
  4. Local adaptation for this repo
- If paper and code disagree, prefer official author code and document the mismatch.
- Keep original hyperparameters, preprocessing, and evaluation protocol unless change is required to run locally.
- Any deviation must be explicit in notes/commit message:
  - what changed
  - why it changed
  - expected impact on metrics
- For each reproduced method, record source links (paper + code repo) in the experiment notes.

## Validation and testing policy
- Required by default:
  - Run only the smallest relevant check for the changed code path.
  - Verify that key experiment metrics still compute correctly.
- Optional unless requested:
  - Full test suites
  - Exhaustive edge-case input validation
  - Enterprise-style reliability/security gates
- If tests are slow or unavailable, provide a short manual verification note.

## Security posture for this repo
- Treat this project as research code, not production deployment.
- Do not introduce formal security frameworks by default.
- Keep only baseline hygiene:
  - no hardcoded secrets
  - no intentional unsafe system operations outside task scope
  - no publishing sensitive data artifacts

## Communication style
- Be concise and action-oriented.
- Suggest pragmatic tradeoffs that reduce experiment time.
- Call out when a request shifts from research prototype to production requirements.
