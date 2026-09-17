# Multi-segment well hydrostatic fix (geomechanical unstructured mesh)

## Symptom

In a geomechanical, isothermal model with a multi-perforation well, the pressure
near the well changed over time **even at zero net flow**, redistributing along the
perforated interval as a `+/-` dipole. Turning gravity off, using a single
perforation, or removing the well all made the effect disappear
(see `Pressure_change_wells_issue.pptx`).

```
      Δpressure, 2D slice (reported)              vertical profile near well
      ┌───────────────────────────┐        depth │
      │            ░░              │          ────┤  ┄┄ hydrostatic (expected)
      │        ▓▓▓▓  ░░░░          │              │ ╱  ← t=0 bulges, then
      │        ▓▓ +  -  ░          │  reservoir → │╱     "relaxes" over time
      │        ▓▓▓▓  ░░░░          │              │╲
      │            ░░              │              │ ╲
      └───────────────────────────┘          ────┤  ┄┄
        spurious dipole at the well
```

## Root cause

A well is discretized as a chain of fluid blocks connected to the reservoir:

```
   head block (well control / BHP)
        │  chain connection  (transmissibility + hydrostatic gravity)
   segment 1 ──┐
        │      ├─ perforation ── reservoir cell   (WI, NO gravity term)
   segment 2 ──┤
        │      ├─ perforation ── reservoir cell
   segment 3 ──┘
```

**Original single-segment model** (all perforations share ONE well block):

```
        head (BHP)
            │
        ┌───┴────┐   one block, one pressure  p_w
        │   p_w  │
        └─┬──┬──┬─┘
    z1 ───┘  │  └─── z3     perforations attach at different depths z1..z3
    z2 ──────┘              with NO gravity term, so p_w cannot equal the
                            hydrostatic p_res(z) at every depth simultaneously
                            → the wellbore short-circuits cells at different
                              depths → gravity-driven crossflow (the dipole).
```

**Multi-segment model** (one segment per perforation, the intended fix): each
segment sits at its perforated cell's depth (so the perforation connection needs
no gravity, `dz = 0`), and consecutive segments carry a hydrostatic gravity term
so the well column reproduces the formation gradient:

```
   head (BHP) ── z_top
     │   grav_rhs = T · g · (z_parent − z_child)
   seg1 ── perf ── res(z1)      dz = 0 at the perforation
     │   grav_rhs …
   seg2 ── perf ── res(z2)
     │
   seg3 ── perf ── res(z3)
```

## The two corrections in this commit

### 1. Well-chain gravity **sign** — `engines/src/conn_mesh.cpp`, `add_wells_mpfa`

The reservoir/discretizer encodes gravity direction (downward) with the sign
`(depth[block_m] − depth[block_p])` (cf. `init_grav_coef`). The well chain used the
**opposite** sign, so the well's hydrostatic gradient *opposed* the formation and
the mismatch was **twice** the hydrostatic column.

```diff
- value_t dz = depth[well_head_idx + p + 1] - depth[well_head_idx + p];
+ value_t dz = depth[well_head_idx + p]     - depth[well_head_idx + p + 1];
  value_t grav_rhs = wells[iw]->well_transmissibility * g_constant * dz;
```

### 2. `well_head_depth` — `darts/reservoirs/unstruct_reservoir_mech.py`, `init_wells`

`mesh.depth` stores the z-centroid (elevation; up = `+`, values are negative here).
The well **head is the top / shallowest** perforation = the **largest** z, so `min`
placed the head/BHP datum at the bottom.

```diff
- w.well_head_depth = float(depth[perf_res_ids].min())
+ w.well_head_depth = float(depth[perf_res_ids].max())
```

## Verification

Zero-drawdown transient (well present, gravity on, hydrostatic init, ~zero net
flow); metric = max spurious `|Δp|` along the perforated column. Reservoir
hydrostatic column here is **2.87 bar** (331.62 → 334.49). Model:
`models/SPE10_mech`, mesh `data_10_10_10`.

| configuration                              | max spurious \|Δp\| |
| ------------------------------------------ | ------------------- |
| single-segment (original)                  | **2.80 bar** (dipole) |
| multi-segment, **before** these two fixes  | 5.64 bar (≈ 2× column, wrong sign) |
| multi-segment, **after** these two fixes   | **0.0014 bar** (≈ 0) |

The corrected multi-segment well holds the formation hydrostatic to ~1 mbar — a
~2000× reduction — while the single-segment model retains the ~2.8 bar dipole with
the two wells deviating in opposite directions (the crossflow signature).

> Reproduce with `models/SPE10_mech/compare_trans.py`.
>
> Note: this build path also needs `cpu_superlu` (the model's default
> `cpu_gmres_fs_cpr` requires the MT/bos-solvers build), and two unrelated
> attribute bugs must be fixed for the model to run unpatched (see the follow-up
> commit): `add_perforation` uses `self.discretizer` (should be `self.discr`) and
> `init_wells` uses `self.n_res_blocks` (should be `self.n_matrix`).
