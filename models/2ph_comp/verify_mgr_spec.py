"""Phase-1 verification: MGRSolverSpec reproduces the raw set_*-call MGR build.

Runs models/2ph_comp twice -- once with the existing raw-object MGR build
(Model.set_solver) and once with an equivalent MGRSolverSpec routed through the
solver registry / new C++ mgr_solver_config fields -- and asserts that the
time-step / Newton / linear iteration counts are identical.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from darts.engines import redirect_darts_output, sim_params  # noqa: E402
from darts.nonlinear_solvers import ChopSpec, NewtonSolver  # noqa: E402
from model import Model  # noqa: E402

from darts.linear_solvers import (  # noqa: E402
    BCSRCPRSpec,
    BILU0Spec,
    LocalCorrectionSpec,
    MGRLevelSpec,
    MGRSolverSpec,
    PressureAMGSpec,
)
from darts.linear_solvers.enums import (  # noqa: E402
    BCSRCPRReduction,
    CoarseGrid,
    CompositeMode,
    FRelaxation,
    GlobalSmoother,
    Interpolation,
    LocalFallback,
    LocalPreconditioner,
    Restriction,
    VariableRole,
)


class ModelSpec(Model):
    """2ph_comp with the raw MGR build replaced by an equivalent MGRSolverSpec."""

    def set_solver(self):
        # Base Model.set_solver() now owns set_sim_params(); replicate it here since
        # this override does not call super().
        self.linear_solver.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000)
        super().set_solver()  # platform default nonlinear + linear solvers
        # must mirror Model.set_solver()'s nonlinear settings exactly -- this
        # harness compares the raw MGR build against the spec build, so any
        # nonlinear difference would show up as a bogus iteration-count divergence
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=20,
                                             chop=ChopSpec(mode='local'))
        self.params.linear_print_level = 0
        block_size = self.physics.n_vars
        mesh = getattr(self.reservoir, "mesh", None)
        reservoir_blocks = mesh.n_res_blocks if mesh is not None else 0

        reservoir_roles = [VariableRole.PRESSURE] + [VariableRole.COMPOSITION] * (
            block_size - 1
        )
        well_roles = [VariableRole.WELL_PRESSURE] + [VariableRole.WELL_SECONDARY] * (
            block_size - 1
        )

        self._solver_spec = MGRSolverSpec(
            tolerance=1e-4,
            max_iterations=50,
            log_level=self.params.linear_print_level,
            kdim=150,
            use_mgr=True,
            use_flex_gmres=True,
            use_physics_scaling=True,
            composite_mode=CompositeMode.MGR_THEN_LOCAL,  # 1
            local_solver=LocalPreconditioner.BLOCK_ILU0,  # 2
            bilu0=BILU0Spec(
                pivot_shift=1e-12,
                fallback_strategy=LocalFallback.IDENTITY,  # 0
                fallback_diagonal_tolerance=1e-4,
                fallback_shifted_max=1e-4,
                fallback_shifted_growth=100.0,
            ),
            local_correction=LocalCorrectionSpec(
                alpha=1.0,
                adaptive_fallback_threshold=-1.0,
                adaptive_alpha=0.0,
                adaptive_fallback_threshold_high=-1.0,
                adaptive_alpha_high=0.0,
                quality_enabled=False,
                quality_min_alpha=0.0,
            ),
            pressure_amg=PressureAMGSpec(
                coarsen_type=6,
                interp_type=6,
                relax_type=6,
                agg_num_levels=1,
                agg_interp_type=6,
                agg_pmax_elmts=20,
                relax_order=1,
                strong_threshold=0.5,
                trunc_factor=-1.0,
                pmax_elmts=-1,
                max_levels=0,
                solve_max_iter=1,
                solve_tolerance=0.0,
            ),
            bcsr_cpr=BCSRCPRSpec(
                reduction_type=BCSRCPRReduction.TRUE_IMPES,  # 1
                pressure_variable=0,
                weight_max=1e6,
                reuse_amg_hierarchy=True,
                amg_rebuild_interval=0,
                adaptive_amg_rebuild=True,
                adaptive_li_threshold=15,
                adaptive_li_growth_factor=1.5,
                adaptive_min_reuse_setups=1,
                adaptive_max_reuse_setups=2,
                adaptive_pressure_overshoot_threshold=-1.0,
                adaptive_final_proxy_threshold=-1.0,
                adaptive_fallback_threshold=-1.0,
                diagnostics=True,
                diagnostic_apply_interval=100,
                diagnostic_matrix_interval=0,
                pressure_correction_alpha=1.0,
                pressure_correction_guard_threshold=10.0,
                pressure_correction_guard_min_alpha=0.05,
                # transpose_apply / forward_source left None (auto-derivation)
            ),
            reservoir_variable_roles=reservoir_roles,
            well_variable_roles=well_roles,
            pressure_level=MGRLevelSpec(
                frelax_type=FRelaxation.NONE,  # -1
                frelax_iters=0,
                interp_type=Interpolation.INJECTION,  # 0
                restrict_type=Restriction.BLOCK_COL_LUMPED,  # 14
                coarse_method=CoarseGrid.GALERKIN,  # 0
                smoother_type=GlobalSmoother.HYPRE_ILU,  # 16
                smoother_iters=1,
            ),
            n_reservoir_blocks=int(reservoir_blocks),
            enable_well_level=False,
            enable_composition_level=False,
        )
        # Phase 3: assign the SPEC directly to self.linear_solver. The unified base
        # _apply_solver(stage="pre") builds and injects it BEFORE engine.init --
        # the migration target for 2ph_comp. (Phase 1 instead built the object and
        # injected it post-init; this checks the pre-init spec path is equivalent.)
        self.linear_solver.spec = self._solver_spec
        self.linear_solver.label = "mgr (bcsr-cpr, spec)"


def run_and_collect(model_cls, log):
    redirect_darts_output(log)
    n = model_cls()
    n.init()
    n.set_output()
    n.run(1000)
    st = n.nonlinear_solver.stats
    stats = {
        "n_timesteps_total": st.n_timesteps_total,
        "n_timesteps_wasted": st.n_timesteps_wasted,
        "n_newton_total": st.n_newton_total,
        "n_newton_wasted": st.n_newton_wasted,
        "n_linear_total": st.n_linear_total,
        "n_linear_wasted": st.n_linear_wasted,
    }
    X = np.array(n.physics.engine.X, copy=True)
    return stats, X


if __name__ == "__main__":
    raw_stats, raw_X = run_and_collect(Model, "verify_raw.log")
    spec_stats, spec_X = run_and_collect(ModelSpec, "verify_spec.log")

    print("\n================ Phase-1 MGR spec verification ================")
    print(f"{'metric':22s} {'raw':>12s} {'spec':>12s} {'match':>7s}")
    all_match = True
    for k in raw_stats:
        m = raw_stats[k] == spec_stats[k]
        all_match &= m
        print(f"{k:22s} {raw_stats[k]:>12d} {spec_stats[k]:>12d} {str(m):>7s}")

    dx = float(np.max(np.abs(raw_X - spec_X))) if raw_X.shape == spec_X.shape else np.inf
    print(f"{'max|dX|':22s} {dx:>33.3e}")
    print("===============================================================")
    if all_match and dx < 1e-10:
        print("PASS: MGRSolverSpec reproduces the raw MGR build exactly.")
        sys.exit(0)
    else:
        print("FAIL: spec build diverges from the raw MGR build.")
        sys.exit(1)
