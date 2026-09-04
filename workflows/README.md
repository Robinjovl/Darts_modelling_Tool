# workflows/ — deterministic study layer

Repository-local Python package for the agentic workflows (design:
`docs/for_developers/agentic_workflows_design.md`, version 4). It is imported through the checkout
and is not packaged or installed separately.

```bash
export REPO=/oahu/data/avnovikov/open-darts-skills
PYTHONPATH=$REPO /oahu/data/avnovikov/mambaforge/envs/skills/bin/python -m unittest discover -s $REPO/workflows/tests -v
```

Status (step 2 of the implementation order): `spec.py` (study document, schema, identities, seed
ledger), `journal.py` (study directory, atomic manifests, journal, provenance), `cost.py`
(simulation counts, worker sizing, estimates), `requirements.txt`, the fast test lane, and the
feasibility spike under `evals/spikes/`. Adapters, executor, drivers, CLI and the evaluation
controller follow in later steps.

## Where things are

- Shared reference for the three study skills: `workflows/docs/study-substrate.md`.
- Skills: `.agents/skills/{ensemble-study,history-matching,optimization}-open-darts` (mirrored
  to `.claude/skills`; `helper_scripts/validate_skills.py` checks them).
- Evaluation harness (developer tool, hidden references): `workflows/evals/README.md`.
- Design and evidence: `docs/for_developers/agentic_workflows_design.md`.
