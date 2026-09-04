# Priors and parameterization

Families (`workflows/members.py`): `ScalarParam(low, high)`, `LogScalarParam(low, high)`
(uniform in log10), `MultiplierField(n, sigma_log10, groups)` (log-normal multipliers per group),
`LogPermField(mean_log10, sigma_log10, range_m, n_components)` (Gaussian log-perm field on cell
centroids, Karhunen–Loève truncated to `n_components`; `energy_fraction` is reported).

Targets and their cost on the Brugge proxy (`rebuild_scope`):

| Target | Applies to | Cost |
|---|---|---|
| `tran_multiplier`, `wi_multiplier`, `poro`, `bhp` | mesh transmissibility, well index, porosity, well BHP | cheap: apply + reset |
| `permx` | full permeability field | rebuild (mesh, wells) |
| `well_xyz` | well location by name | rebuild |
| relperm/PVT | not exposed as families yet | new process and OBL points |

Guidance:
- Use log scales for permeability, transmissibility and multipliers; bound multipliers to
  about [0.2, 5] unless the data says otherwise.
- Keep D ≤ 10–15 for Saltelli; use Morris first to drop inert inputs.
- Geology fields: `range_m` about a quarter of the field extent is a reasonable start; more
  components than needed only adds cost, `energy_fraction ≥ 0.9` is a sane check.
- Realization keys must match the adapter's `apply/build`; the substrate rejects unknown
  targets at parameterization time.
