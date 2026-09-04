# workflows/ — deterministic study layer

Repository-local Python package for the agentic workflows (design:
`docs/for_developers/agentic_workflows_design.md`, version 4). It is imported through the checkout
and is not packaged or installed separately.

```bash
export REPO=/oahu/data/avnovikov/open-darts-skills
PYTHONPATH=$REPO /oahu/data/avnovikov/mambaforge/envs/skills/bin/python -m unittest discover -s $REPO/workflows/tests -v
```

Status: implementation steps 2-8 of the design are in place. `spec.py` (study document, schema,
identities, seed ledger), `journal.py` (study directory, atomic manifests, journal, provenance),
`cost.py` (simulation counts, worker sizing, estimates), `adapter.py` + `adapters/brugge_proxy.py`
(model lifecycle; `models/Uniform_Brugge` only), `members.py` (parameter families, log-perm field),
`executor.py` (one process per simulation, walltime, retries, rusage), `ensemble.py` (LHS, Sobol,
Morris, Saltelli; bootstrap percentiles and indices), `esmda.py` (ES-MDA over dageo, diagonal
noise, spatial localization, twin truths, gates), `optimize.py` (exhaustive well placement),
`gates.py`, `cli.py` (`python -m workflows estimate|run|truth|analyze`), `evals/` (sandboxed,
brokered agent evaluation with hidden references). Not implemented: adjoint and derivative-free
drivers, surrogates, robust objectives, full-field Brugge bed, packaging.

## Where things are

- Shared reference for the three study skills: `workflows/docs/study-substrate.md`.
- Skills: `.agents/skills/{ensemble-study,history-matching,optimization}-open-darts` (mirrored
  to `.claude/skills`; `helper_scripts/validate_skills.py` checks them).
- Evaluation harness (developer tool, hidden references): `workflows/evals/README.md`.
- Design and evidence: `docs/for_developers/agentic_workflows_design.md`.
