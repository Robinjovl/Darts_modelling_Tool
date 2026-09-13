# Dynamic poromechanics: time integration of the inertia term

The poromechanics engine `engine_pm_cpu` (unstructured reservoir with the
`pm_discretizer`, contact mechanics by the penalty method with return mapping)
solves the quasi-static momentum balance by default. A run switches to the fully
dynamic momentum balance $\rho\,\ddot{\mathbf u} - \nabla\cdot\boldsymbol\Sigma
- \rho g\nabla z = 0$ by setting `engine.momentum_inertia` to the total density
$\rho$ (kg/m$^3$); `0` keeps the quasi-static form. This is how the
`displaced_fault_reactivation` model resolves the co-seismic phase of a fault
reactivation: quasi-static until the Newton loop fails near the nucleation point,
then dynamic steps of a few hundred microseconds.

The inertia term is integrated in time by one of the schemes below, selected on the
engine (`engine.time_integration`, an enum `darts.engines.time_integration`) or,
more conveniently, through
`darts.nonlinear_solvers.configure_time_integration(engine, scheme, **params)`.
All schemes are implicit, unconditionally stable (for the parameter ranges
given) and formulated with the displacements as the only unknowns, so the
contact (fault) rows, the flow rows, the Newton update and the linear solvers are
untouched; the quasi-static path is bit-identical to previous releases.

| `scheme` | Parameters | Order | Numerical dissipation | Notes |
|---|---|---|---|---|
| `backward_euler` (default) | – | 1 | very strong | legacy 3-point backward scheme on $u^{n-1}, u^n, u^{n+1}$ (`engine.Xn1`, `engine.dt1`); smears wave fronts (thesis Fig. 6.1) |
| `newmark` | `gamma` (0.5), `beta` (0.25) | 2 for $\gamma = 1/2$, else 1 | none for $\gamma = 1/2$; first-order damping for $\gamma > 1/2$, $\beta = (\gamma + 1/2)^2/4$ | Newmark-$\beta$ in displacement form (Chopra 2012; Jacobsen et al. 2024): $\ddot u^{n+1} = [u^{n+1} - u^n - \Delta t\,\dot u^n - \Delta t^2(\tfrac12 - \beta)\ddot u^n]/(\beta\Delta t^2)$, $\dot u^{n+1} = \dot u^n + \Delta t[(1-\gamma)\ddot u^n + \gamma\ddot u^{n+1}]$; unconditionally stable for $\gamma \ge 1/2$, $\beta \ge \gamma/2$ |
| `generalized_alpha` | `rho_inf` (0.8) | 2 | high-frequency, controlled by $\rho_\infty \in [0, 1]$ | Chung & Hulbert (1993): $\alpha_m = \frac{2\rho_\infty - 1}{\rho_\infty + 1}$, $\alpha_f = \frac{\rho_\infty}{\rho_\infty + 1}$, $\gamma = \tfrac12 - \alpha_m + \alpha_f$, $\beta = \tfrac14(1 - \alpha_m + \alpha_f)^2$; residual $\rho V[(1-\alpha_m)\ddot u^{n+1} + \alpha_m \ddot u^n] + (1-\alpha_f)F(u^{n+1}) + \alpha_f F(u^n)$ with $F(u^n)$ taken from the stored converged fluxes of the previous step |
| `hht` | `alpha` (−0.05), in $[-1/3, 0]$ | 2 | high-frequency | Hilber–Hughes–Taylor (1977) = generalized-$\alpha$ with $\alpha_m = 0$, $\alpha_f = -\alpha$ |
| `bathe` | `bathe_gamma` (0.5) | 2 | high-frequency, strong | Bathe (2007) composite scheme: trapezoidal rule over $\gamma\Delta t$, then the 3-point backward formula over the full step; two linear solves per step. Driven automatically as two engine sub-steps by `MechanicsNewtonSolver.run_timestep` (hook `on_substep` for time-dependent boundary data) |

Common option `kv_damping` = $q \ge 0$ (default 0): Kelvin–Voigt (stiffness-proportional)
artificial viscosity that adds $q\,K(u^{n+1} - u^n)$, i.e. $\eta_{KV} K \dot u$ with
$\eta_{KV} = q\,\Delta t$, to the momentum balance of the matrix cells (the practice of
dynamic-rupture codes, Day et al. 2005, where $q \approx 0.1$–$0.25$). It damps
grid-scale wavelengths irrespective of the time step and is the stabilizer that
removes the ringing behind sharp fronts (see below); it is first order in $\Delta t$ and
must be kept small ($q \lesssim 0.25$) to avoid smearing. Since $\eta_{KV} = q\,\Delta t =
(q\,\mathrm{CFL})\,\Delta z/c_p$, the damping is a grid-scale one when $q\,\mathrm{CFL} \approx 0.07$–$0.18$:
on the 1D benchmark this range (q = 0.1–0.25 at CFL 0.7, 0.05–0.1 at CFL 1.4, 0.25–0.5 at CFL 0.35)
brings the wake below 1–5 % of the pulse amplitude at the cost of a 15–35 % amplitude loss of a
7-cell step pulse; larger $q$ only smears.

The velocity and acceleration of the last converged step are available as
`engine.vel` and `engine.acc` (m/day, m/day$^2$; `ND * n_blocks`, matrix cells) and
`v_x, v_y, v_z` are written to the VTK output of dynamic runs. When a run switches
from quasi-static to dynamic mode the state is (re)initialised from rest by
`configure_time_integration` (`engine.reset_dynamic_state()`); a converged
quasi-static state has zero residual, so zero initial acceleration is consistent.
Variable time steps need no history beyond the previous level (except for the legacy
backward Euler scheme, whose 3-point formula is consistent only for
$\Delta t_n = \Delta t_{n-1}$), and a failed step leaves the velocity/acceleration
state untouched.

## Which scheme to use

Benchmark `models/elastic_wave_1d` (thesis Sec. 6.4.1): a 10 m elastic column,
$E = 1$ GPa, $\nu = 0.25$, $\rho = 2406$ kg/m$^3$, rollers, a rectangular
compression pulse of −0.01 m lasting 1 ms prescribed at the top; d'Alembert
solution for comparison. With 100 cells and CFL $= c_p\Delta t/\Delta z = 0.706$
(`python compare_schemes.py`):

| scheme | L2 error (rel., t = 12 ms) | overshoot (t = 2 ms) | wake behind the pulse (t = 12 ms) |
|---|---|---|---|
| backward Euler | 0.75 | 0 | 0.07 |
| Newmark trapezoidal | 0.70 | 0.29 | 0.18 |
| Newmark $\gamma = 0.6$ | 0.56 | 0.17 | 0.03 |
| HHT $\alpha = -0.3$ / generalized-$\alpha$ $\rho_\infty = 0.5$ | 0.69 | 0.25 | 0.14 |
| generalized-$\alpha$ $\rho_\infty = 0$ | 0.75 | 0.13 | 0.24 |
| Bathe | 0.59 | 0.27 | 0.16 |
| trapezoidal + Kelvin–Voigt $q = 0.25$ | 0.56 | 0.06 | $< 10^{-3}$ |
| generalized-$\alpha$ 0.5 + $q = 0.1$ | 0.58 | 0.15 | 0.04 |
| Bathe + $q = 0.1$ | 0.51 | 0.20 | 0.05 |

(overshoot and wake relative to the pulse amplitude.)

![All methods against the propagating step wave](images/elastic_wave_1d_step_wave.png)

*All schemes against the analytical step wave at t = 2 ms and 12 ms (`compare_schemes.py`,
`models/elastic_wave_1d/figures/step_wave_comparison.png`).*

The non-dissipative
trapezoidal rule reproduces the "multiple spurious oscillations across the domain"
of the earlier attempt (thesis Fig. 6.2). The $\alpha$-schemes damp only the modes with
$\omega\Delta t \gg 1$; on a lumped-mass finite-volume grid the ringing behind a step
front lives at $\omega\Delta t \approx 2\,\mathrm{CFL}$, so at CFL $\lesssim 1$ it is a
spatial (dispersion) artefact that no second-order time scheme removes by itself.
Stiffness-proportional damping does remove it. Recommended for co-seismic fault
runs: `{'scheme': 'generalized_alpha', 'rho_inf': 0.5, 'kv_damping': 0.1}`
(one solve per step, annihilates the stiff contact-penalty modes, no wake) or
`{'scheme': 'bathe', 'kv_damping': 0.1}` (lowest error, two solves per step). Keep
`backward_euler` for regression references.

In the displaced-fault model the scheme is a configuration entry:

```python
config = {'mode': 'mixed', 'depletion': {'mode': 'well', 'value': -250.0}, 'friction_law': 'slip_weakening',
          'mesh_file': 'meshes/new_setup_coarse.geo',
          'time_integration': {'scheme': 'generalized_alpha', 'rho_inf': 0.5, 'kv_damping': 0.1}}
```

and `models/displaced_fault_reactivation/dynamic_benchmark.py` runs the mixed
quasi-static/dynamic case for a chosen scheme with per-step diagnostics of the
fault response (slip area, slip rate, seismic moment, traction roughness, Newton
counts).

## Verification (1D benchmark)

- **Order of accuracy** (smooth raised-cosine pulse, joint refinement at CFL 0.706,
  nz = 50…400): trapezoidal Newmark, generalized-α (ρ∞ = 0.8, 0.5), HHT (α = −0.05)
  and Bathe converge at 1.8 in L2, which is the rate of the spatial discretization
  (fixed-mesh temporal refinement gives 1.91–1.94 after removing the spatial error
  floor, and all three reach the same floor to 0.7 %); Newmark γ = 0.6 and the
  Kelvin–Voigt-damped trapezoidal rule are first order (q = 0.1 is dynamically
  equivalent to γ = 0.6: both have damping ratio 0.05 ω Δt to leading order);
  backward Euler is far from its asymptotic regime (observed 0.2–0.7) because it
  loses 16–70 % of the pulse amplitude. Bathe has the smallest error constant
  (0.77× trapezoidal) at twice the cost per step.
- **Discrete energy** (kinetic + strain, monitored through two reflections): the
  trapezoidal rule conserves it to machine precision (ratio 1.0000, largest
  step-wise increase 7e-16); every other scheme dissipates monotonically (no
  step ever increases the energy); Newton converges in one iteration per solve
  for this linear problem.
- **State handling**: a forced Newton failure rolls back to the last converged
  state without touching the velocity/acceleration state, and the resumed run is
  bit-identical to an uninterrupted one for every scheme (including a failure in
  either Bathe sub-step); switching a quasi-static run to dynamic mode gives the
  same bits as a run started dynamic.
- **Bathe consistency**: the sub-step identities (trapezoidal sub-step,
  three-point backward sub-step, rotation of `Xn1/Xn` and `vel_n1/vel`) hold to
  1e-14 in a quasi-static ramp test at 11 and 114 steps per period of the column's
  fundamental mode; Bathe is the scheme closest to a converged dynamic reference,
  while backward Euler damps and delays the transient (26 % at 114 steps per
  period). Bathe's Kelvin–Voigt viscosity acts per sub-step (η = q γ Δt), so a
  Bathe run needs q ≈ 1.5× that of a single-step scheme for the same damping.
- **Step response versus CFL** (rectangular pulse, CFL 0.35–4): Newmark γ = 0.6 gives
  the cleanest front per linear solve at every CFL (zero overshoot, wake 3–8× below
  Bathe, 10–25 % amplitude loss); Bathe is cleaner than the trapezoidal rule at the
  same Δt with a mild optimum at CFL 1–1.4, but on a 7–14-cell pulse the spatial
  dispersion dominates, so the literature claim of a clean step response needs
  ≥ 20–40 cells per pulse.

## Co-seismic fault runs

`dynamic_benchmark.py` (mixed run, well depletion, coarse mesh, 1500 dynamic
steps of 0.5 ms) shows the same rupture sequence for every scheme — nucleation on
the two reservoir-corner patches, rupture of the whole fault within ~0.15 s, arrest
with ~0.4 m of slip and peak slip rates of ~10 m/s (consistent with a 10–20 MPa
stress drop and the shear impedance G/2c_s ≈ 2 MPa·s/m) — with backward Euler
lagging the second-order schemes by 20–30 ms, as in the 1D ramp test. Two
practical points:

- The Newmark-family Jacobians carry the mass term ρV/(βΔt²), four times larger
  than backward Euler's ρV/Δt², so GMRES + FS-CPR needs about a third of the
  iterations (25–30 versus 75 per Newton iteration on the coarse mesh).
- At the rupture front the penalty return mapping can chatter between stick and
  slip on successive Newton iterations. Backward Euler escapes by cutting Δt (a
  frozen matrix makes the constraint trivially satisfiable), but a velocity-state
  scheme cannot be relaxed that way: the free-flight motion u^n + Δt v^n is imposed
  at any Δt, and below ~1e-6 s the 1/(βΔt²) amplification of displacement
  round-off floors the momentum residual at ~1e-4. The driver therefore redoes a
  dynamic step that failed after two cuts with backward Euler and returns to the
  selected scheme afterwards (`run_python` in `displaced_fault_reactivation/main.py`;
  the count is reported as `n_be_fallbacks`). A state-freezing or line-search
  strategy inside the contact Newton loop would be the systematic cure.

## Implementation notes

- Assembly (`engine_pm_cpu::assemble_jacobian_array` and the time-dependent
  discretization variant): for matrix cells the inertia residual
  $\rho V[(1-\alpha_m)\ddot u^{n+1}(u^{n+1}) + \alpha_m\ddot u^n]$ is added with the
  diagonal Jacobian $\rho V(1-\alpha_m)\,\partial\ddot u^{n+1}/\partial u^{n+1}$
  ($= 1/(\beta\Delta t^2)$ for the Newmark family, $c_3^2$ for the second Bathe
  sub-step); the internal-force Jacobian entries of those rows are scaled by
  $(1-\alpha_f) + q$ (displacement columns) and $(1-\alpha_f)$ (pressure column),
  and the residual gets $\alpha_f[F(u^n) - F(u^{n+1})] + q\,K(u^{n+1}-u^n)$. Fault
  (contact) rows are overwritten by the contact assembly and stay fully implicit
  constraints at $t^{n+1}$; well rows are untouched. Units: displacement m,
  pressure bar, time days (`BAR_DAY2_TO_PA_S2` converts $\rho V \ddot u$ to bar·m$^2$).
- `post_newtonloop` commits $\dot u^{n+1}, \ddot u^{n+1}$ (`commit_dynamic_state`)
  before rotating `Xn`; a rejected step changes nothing.
- The Bathe second sub-step uses `Xn1`/`vel_n1` (level $n$) and `Xn`/`vel`
  (level $n+\gamma$); a failed second sub-step leaves the engine at
  $t^{n+\gamma}$ and the driver simply continues from there.
- `models/elastic_wave_1d` uses a column two cells wide: on a one-cell-wide column
  the roller side faces make the in-plane gradient reconstruction of
  `pm_discretizer` degenerate ($\sim 10^{30}$ entries in the $u_x, u_y$ rows).

References: Newmark (1959); Hilber, Hughes & Taylor (1977); Chung & Hulbert (1993);
Bathe (2007), Bathe & Noh (2012); Day, Dalguer, Lapusta & Liu (2005); Jacobsen,
Berre, Nordbotten & Stefansson (2024, MPSA–Newmark); Novikov (PhD thesis, TU Delft,
Ch. 6).
