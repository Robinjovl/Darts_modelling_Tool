"""FD consistency tests for the wellhead-control Jacobian row (M1b).

``engines/src/well_controls.cpp`` replaces the wellhead equation row with the
control residual and its derivatives; the fix under test places the rate-
control operator derivatives in the column block of the STATE that was used
(head block for injectors, body block for producers), plus the product-rule
pressure terms in both pressure columns. These tests verify the whole row by
finite differences against the assembled system:

* build a tiny 5-cell EPM model (test-local class, darts imports only --
  ConstantK flash, basic densities, adapted from ``models/2ph_comp``);
* per control case (BHP, total MOLAR/MASS rate, phase MOLAR rate; injector
  and producer): set the controls, ``engine.assemble_linear_system(dt)``,
  extract the wellhead-row diagonal and head->body column blocks through
  ``BlockCSRView``;
* perturb each state variable of the wellhead and body blocks (h chosen per
  variable scale), re-assemble, and compare the FD of the wellhead RHS rows
  against the corresponding Jacobian column (rtol 1e-4). The OBL operators
  are piecewise-multilinear, so within an interpolation cell the one-sided FD
  is exact up to rounding; perturbing X only ADDS adaptive interpolation
  points, so the check is deterministic.

Also locked in: the intentional degenerate-phase regularization (an absent
target phase puts an exact 1.0 on the wellhead pressure diagonal so the
pressure can fall until the phase appears -- well_controls.cpp, "if target
phase does not exist"), and the DFM-well rejection path of
``LinearDFMWellIPRHook`` (EPM wells are not supported).

The DFM-well variant of the rate-control row (which consumes the Python pipe
loop's ``phases_vels``) is not exercised here -- see the skip marker at the
bottom.
"""

import numpy as np
import pytest

pytest.importorskip("darts.engines")

from darts.engines import well_control_iface  # noqa: E402
from darts.models.conditions import BlockCSRView  # noqa: E402
from darts.models.darts_model import DartsModel  # noqa: E402
from darts.nonlinear_solvers import ChopSpec, NewtonSolver  # noqa: E402
from darts.physics.base.physics import PhysicsBase  # noqa: E402
from darts.physics.base.property_container import PropertyContainer  # noqa: E402
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm  # noqa: E402
from darts.physics.properties.density import DensityBasic  # noqa: E402
from darts.physics.properties.flash import ConstantK  # noqa: E402
from darts.reservoirs.struct_reservoir import StructReservoir  # noqa: E402

DT = 0.1  # days
INJ_COMPOSITION = [1.0 - 2e-7, 1e-7]  # nearly pure CO2


class _TinyEPMModel(DartsModel):
    """5-cell 1D two-phase three-component model with one injector (cell 1)
    and one producer (cell 5); a shrunk copy of ``models/2ph_comp``."""

    def __init__(self):
        super().__init__()
        self.set_reservoir()
        self.set_physics()
        self.nonlinear_solver = NewtonSolver(
            tolerance=1e-2, max_iterations=10, chop=ChopSpec(mode="local")
        )
        self.data_ts.dt_first = 0.001
        self.data_ts.dt_mult = 2.0
        self.data_ts.dt_max = 1.0
        self.data_ts.linear_tol = 1e-4
        self.data_ts.linear_max_iter = 50

    def set_reservoir(self):
        self.reservoir = StructReservoir(
            self.timer,
            nx=5,
            ny=1,
            nz=1,
            dx=10,
            dy=10,
            dz=10,
            permx=100,
            permy=100,
            permz=10,
            poro=0.3,
            depth=1000,
        )

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(5, 1, 1))

    def set_physics(self):
        zero = 1e-8
        epsilon = 1e-9
        components = ["CO2", "C1", "H2O"]
        phases = ["gas", "aqueous"]
        property_container = PropertyContainer(
            phases_name=phases,
            components_name=components,
            Mw=[44.01, 16.04, 18.015],
            eps_z=epsilon,
            temperature=1.0,
        )
        property_container.flash_ev = ConstantK(len(components), [4, 2, 1e-1], zero)
        property_container.density_ev = dict(
            [
                ("gas", DensityBasic(compr=1e-3, dens0=200)),
                ("aqueous", DensityBasic(compr=1e-5, dens0=600)),
            ]
        )
        property_container.viscosity_ev = dict(
            [("gas", ConstFunc(0.05)), ("aqueous", ConstFunc(0.5))]
        )
        property_container.rel_perm_ev = dict(
            [("gas", PhaseRelPerm("gas")), ("aqueous", PhaseRelPerm("oil"))]
        )
        p_step = (300 - 1) / (200 - 1)
        z_step = (1 - 3 * epsilon) / (200 - 1)
        self.physics = PhysicsBase(
            components,
            phases,
            self.timer,
            state_spec=PhysicsBase.StateSpecification.P,
            axes_step=[p_step, z_step, z_step],
            axes_origin=[1.0, epsilon, epsilon],
            epsilon_z=epsilon,
        )
        self.physics.add_property_region(property_container)

    def set_initial_conditions(self):
        # p = 50 bar, z = (0.1, 0.2): a single-phase (aqueous) interior state
        # away from the OBL cell faces, so small FD steps stay in one cell
        input_distribution = {
            self.physics.vars[0]: 50,
            self.physics.vars[1]: 0.1,
            self.physics.vars[2]: 0.2,
        }
        return self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=input_distribution
        )

    def set_well_controls(self):
        for well in self.reservoir.wells:
            if "I" in well.name:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=140.0,
                    inj_composition=INJ_COMPOSITION,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=45.0,
                )


@pytest.fixture(scope="module")
def epm_model():
    model = _TinyEPMModel()
    model.init(platform="cpu", verbose=0)
    return model


def _set_control(model, well, control_type, is_inj, target, phase_name=None):
    model.physics.set_well_controls(
        wctrl=well.control,
        control_type=control_type,
        is_inj=is_inj,
        target=target,
        phase_name=phase_name,
        inj_composition=INJ_COMPOSITION if is_inj else None,
    )


def _wellhead_row_fd_check(model, well, steps=(3e-5, 1e-7, 1e-7)):
    """Assemble, extract the wellhead row blocks, FD every column of the head
    and body blocks, and assert J * dX ~= dRHS on the wellhead rows."""
    engine = model.physics.engine
    n_vars = model.physics.n_vars
    view = BlockCSRView(engine, n_vars)
    X = np.asarray(engine.X)
    head, body = well.well_head_idx, well.well_body_idx
    head_rows = slice(head * n_vars, (head + 1) * n_vars)

    engine.assemble_linear_system(DT)
    rhs_base = np.asarray(engine.RHS)[head_rows].copy()

    def block(pos):
        start = pos * view.block_size
        return (
            view.jac_vals[start : start + view.block_size]
            .reshape(n_vars, n_vars)
            .copy()
        )

    J_head = block(view.diag_pos(head))
    J_body = block(view.block_pos(head, body))

    for target_block, J in ((head, J_head), (body, J_body)):
        for var in range(n_vars):
            h = steps[var]
            idx = target_block * n_vars + var
            x_saved = X[idx]
            X[idx] += h
            engine.assemble_linear_system(DT)
            fd_col = (np.asarray(engine.RHS)[head_rows] - rhs_base) / h
            X[idx] = x_saved

            j_col = J[:, var]
            scale = max(np.abs(j_col).max(), np.abs(fd_col).max())
            if scale == 0.0:
                continue  # both identically zero: consistent
            np.testing.assert_allclose(
                fd_col,
                j_col,
                rtol=1e-4,
                atol=1e-6 * scale,
                err_msg=(
                    f"{well.name} wellhead row, "
                    f"{'head' if target_block == head else 'body'} column {var}"
                ),
            )
    # restore a clean assembly at the base state for the next case
    engine.assemble_linear_system(DT)
    return J_head, J_body


CONTROL_CASES = [
    ("bhp", well_control_iface.BHP, 140.0, 45.0, None),
    ("total_molar_rate", well_control_iface.MOLAR_RATE, 20.0, 20.0, None),
    ("total_mass_rate", well_control_iface.MASS_RATE, 500.0, 500.0, None),
    # phase rate on the phase PRESENT at this state (aqueous); the absent-phase
    # branch is covered by test_absent_phase_rate_regularization below
    ("aqueous_molar_rate", well_control_iface.MOLAR_RATE, 5.0, 5.0, "aqueous"),
]


@pytest.mark.parametrize(
    "control_type,inj_target,prod_target,phase_name",
    [case[1:] for case in CONTROL_CASES],
    ids=[case[0] for case in CONTROL_CASES],
)
def test_wellhead_row_jacobian_matches_fd(
    epm_model, control_type, inj_target, prod_target, phase_name
):
    injector, producer = epm_model.reservoir.wells
    # injector phase-rate targeting the injected (gas-like) stream is skipped:
    # at this state the head state holds no gas, hitting the degenerate branch
    inj_phase = None if phase_name == "aqueous" else phase_name
    _set_control(
        epm_model, injector, control_type, True, inj_target, phase_name=inj_phase
    )
    _set_control(
        epm_model, producer, control_type, False, prod_target, phase_name=phase_name
    )
    for well in (injector, producer):
        _wellhead_row_fd_check(epm_model, well)


def test_bhp_row_is_identity_on_wellhead_pressure(epm_model):
    """BHP control: residual p_head - target, so the head diagonal block's
    pressure row is exactly the unit vector and the body column block is 0
    in that row."""
    injector, producer = epm_model.reservoir.wells
    _set_control(epm_model, injector, well_control_iface.BHP, True, 140.0)
    _set_control(epm_model, producer, well_control_iface.BHP, False, 45.0)
    for well in (injector, producer):
        J_head, J_body = _wellhead_row_fd_check(epm_model, well)
        assert J_head[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
        assert J_body[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_absent_phase_rate_regularization(epm_model):
    """Producer gas-rate control at a gas-free state: the rate residual has a
    ~zero pressure derivative, and well_controls.cpp intentionally places an
    exact 1.0 on the wellhead pressure diagonal ("it will let the pressure
    drop and eventually pressure constraint might work"). The FD of the
    residual is legitimately ~0 here, so this asserts the placed value, not
    FD consistency."""
    injector, producer = epm_model.reservoir.wells
    _set_control(epm_model, injector, well_control_iface.BHP, True, 140.0)
    _set_control(
        epm_model,
        producer,
        well_control_iface.MOLAR_RATE,
        False,
        5.0,
        phase_name="gas",
    )
    engine = epm_model.physics.engine
    n_vars = epm_model.physics.n_vars
    engine.assemble_linear_system(DT)
    view = BlockCSRView(engine, n_vars)
    head = producer.well_head_idx
    start = view.diag_pos(head) * view.block_size
    J_head = view.jac_vals[start : start + view.block_size].reshape(n_vars, n_vars)
    assert J_head[0, 0] == 1.0
    # restore BHP control and a clean assembly for other tests
    _set_control(epm_model, producer, well_control_iface.BHP, False, 45.0)
    engine.assemble_linear_system(DT)


def test_linear_dfm_well_ipr_hook_rejects_epm_wells(epm_model):
    """LinearDFMWellIPRHook supports DFM wells only; the EPM wells of this
    model must be rejected with NotImplementedError."""
    from darts.pipes.linear_dfm_well_ipr import (
        LinearDFMWellIPRConnection,
        LinearDFMWellIPRHook,
        PI_Type,
    )

    hook = LinearDFMWellIPRHook(
        epm_model,
        [
            LinearDFMWellIPRConnection(
                well_name="P1", perforation_index=0, pi=1.0, pi_type=PI_Type.MASS
            )
        ],
    )
    view = BlockCSRView(epm_model.physics.engine, epm_model.physics.n_vars)
    with pytest.raises(NotImplementedError, match="DFM"):
        hook._get_resolved_connections(view)


@pytest.mark.skip(
    reason=(
        "DFM-well rate-control rows consume phases_vels/phases_vels_ders from "
        "the Python pipe loop (model.update_dfm_well_vels_and_ders); driving "
        "that loop to a two-phase flowing state deterministically requires "
        "running timesteps of the coupled OLGA model, which is out of scope "
        "for this FD unit suite. The EPM injector/producer BHP and rate rows "
        "above cover the well_controls.cpp column-placement fix."
    )
)
def test_dfm_wellhead_rate_row_fd():
    pass
