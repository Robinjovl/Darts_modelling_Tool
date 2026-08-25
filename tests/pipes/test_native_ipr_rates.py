"""End-to-end rate EXPORT tests for the engine-side (native) linear IPR (R1).

The native perforation flow law (M5) is assembled into the residual/Jacobian
with a ZERO geometric well index, so both rate-reporting paths -- the modern
``Output.store_well_time_data`` (operators * WI * dp) and the legacy
``ms_well::calc_rates`` (p_diff * wi) -- used to export every perforation and
summed-perforation rate of a converged native-IPR simulation as exactly 0.0
(review finding R1). Both paths are now law-aware, sharing one source of truth:

* assembly and the C++ reporting both call ``perforation_law_rates()``
  (ms_well.cpp), so what is reported IS what was assembled;
* the Python export mirrors that arithmetic through the same interpolated
  operator table, and is PINNED against the C++ path here
  (:func:`test_cpp_and_python_reporting_agree_bit_for_bit`).

The tests run the real benchmark models (imported by path, exactly like
``test_engine_perforation_flow_law.py``) with the native ``ipr_engine_*``
formulations for a stretch of simulation time, then assert on the exported
``well_*`` time series:

* every law-perforation rate family is exported nonzero with the sign of the
  actual flow direction (positive = injection, the Darcy export convention);
* the exported rate matches the law flux computed from the exported drawdown --
  ``A + B * (p_well - p_res - offset)`` -- at machine precision, in the law's
  own basis (mass for the thermal injector, MOLAR for the producer engine
  formulation, VOLUMETRIC with nonzero intercept/offset for the non-molar
  volumetric formulation);
* summed perforation rates equal the perforation rates, and at steady state
  the material balance closes: the injector's summed perforation mass rate
  equals the wellhead mass-rate control target.

The Darcy branch of both reporting paths is untouched by the law dispatch, so
every existing (Python-hook) CI reference stays byte-identical.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("darts.engines")
pytest.importorskip("dartsflash")

from darts.engines import (  # noqa: E402
    ms_well,
    redirect_darts_output,
    value_vector,
    well_control_iface,
)
from darts.pipes.linear_dfm_well_ipr import LinearIPR, PI_Type  # noqa: E402
from darts.tools.hdf5_tools import load_hdf5_to_dict  # noqa: E402

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "dfm_well"
ISOTHERMAL_DIR = MODELS_DIR / "2ph_2comp_isothermal_dfm_vertical_well_vs_olga"
THERMAL_DIR = MODELS_DIR / "1ph_1comp_thermal_dfm_well_vs_olga"

#: the thermal model's wellhead injection control target [kg/day]
#: (``target_inj_rate = 2 * 24 * 3600`` in its ``set_well_controls``)
THERMAL_INJ_TARGET = 2.0 * 24 * 3600
#: the thermal model's perforation mass PI [kg/day/bar] (``IPR_PRODUCTIVITY``)
THERMAL_MASS_PI = 1.0e5


def _load_model_class(model_dir: pathlib.Path, module_name: str):
    if not model_dir.is_dir():
        pytest.skip(f"model directory not found: {model_dir}")
    sys.path.insert(0, str(model_dir))
    sys.path.insert(0, str(MODELS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, model_dir / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODELS_DIR))
        sys.path.remove(str(model_dir))
    return module.Model


def _run(model, seconds: float, output_folder) -> dict:
    """Initialize, run ``seconds`` of simulation, export the well time data."""
    redirect_darts_output("")
    model.init(platform="cpu", verbose=0)
    model.set_output(output_folder=str(output_folder))
    model.run(seconds / (24.0 * 3600.0))
    return model.output.store_well_time_data()


def _perforation_drawdown(model, well_name: str = "I1") -> np.ndarray:
    """``p_well_segment - p_reservoir`` at perforation 0, per exported time."""
    h5 = load_hdf5_to_dict(model.output.well_filepath)
    well = model.reservoir.get_well(well_name)
    well_block = int(well.well_body_idx) + int(well.perforations[0][0])
    res_block = int(well.perforations[0][1])
    cell_id = h5["dynamic"]["cell_id"]
    iw = int(np.where(cell_id == well_block)[0][0])
    ir = int(np.where(cell_id == res_block)[0][0])
    X = h5["dynamic"]["X"]
    return X[:, iw, 0] - X[:, ir, 0]


# ---------------------------------------------------------------------------
# fixtures: one full model run each, shared by all assertions on it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def thermal_run(tmp_path_factory):
    """The 1ph thermal injector, native MASS-basis IPR, run to steady state
    (the full 10-minute benchmark runtime, ~300 timesteps)."""
    cls = _load_model_class(THERMAL_DIR, "native_ipr_rates_thermal_model")
    model = cls(formulation="ipr_engine")
    td = _run(model, 600.0, tmp_path_factory.mktemp("native_ipr_thermal"))
    return model, td


@pytest.fixture(scope="module")
def producer_run(tmp_path_factory):
    """The 2ph isothermal BHP producer, native MOLAR-basis IPR."""
    cls = _load_model_class(ISOTHERMAL_DIR, "native_ipr_rates_producer_model")
    model = cls(formulation="ipr_engine_producer")
    td = _run(model, 60.0, tmp_path_factory.mktemp("native_ipr_producer"))
    return model, td


@pytest.fixture(scope="module")
def volumetric_run(tmp_path_factory):
    """The 2ph isothermal injector with a native VOLUMETRIC-basis IPR carrying
    a nonzero intercept and pressure offset (the non-molar-basis case)."""
    cls = _load_model_class(ISOTHERMAL_DIR, "native_ipr_rates_volumetric_model")
    model = cls(formulation="ipr_engine_volumetric")
    td = _run(model, 60.0, tmp_path_factory.mktemp("native_ipr_volumetric"))
    return model, td


# ------------------------------------------------ thermal injector (MASS PI)


def test_injector_exports_every_rate_family_nonzero_and_positive(thermal_run):
    """The review's failure mode: every ``well_I1_perf_0_*`` and
    ``*_by_sum_perfs`` rate of a converged native-IPR run exported as exactly
    0.0. All families must now be nonzero, positive for injection."""
    _, td = thermal_run
    positive = [
        "well_I1_perf_0_mass_rate_CO2",
        "well_I1_perf_0_mass_rate_G",
        "well_I1_perf_0_molar_rate_CO2",
        "well_I1_perf_0_molar_rate_G",
        "well_I1_perf_0_volumetric_rate_G",
        "well_I1_mass_rate_CO2_by_sum_perfs",
        "well_I1_mass_rate_G_by_sum_perfs",
        "well_I1_molar_rate_CO2_by_sum_perfs",
        "well_I1_molar_rate_G_by_sum_perfs",
        "well_I1_volumetric_rate_G_by_sum_perfs",
    ]
    for key in positive:
        assert td[key][-1] > 0.0, f"{key} must be positive for the injector"
    # the energy the law carries into the reservoir is exported too
    assert td["well_I1_perf_0_advective_heat_rate_G"][-1] != 0.0
    assert td["well_I1_advective_heat_rate_G_by_sum_perfs"][-1] != 0.0


def test_mass_rate_is_the_productivity_times_the_drawdown(thermal_run):
    """MASS-basis law: exported mass rate == PI * (p_well - p_res) at every
    exported time, at machine precision (the operator-mixture molecular weight
    the conversion round-trips through cancels exactly)."""
    model, td = thermal_run
    drawdown = _perforation_drawdown(model)
    expected = THERMAL_MASS_PI * drawdown
    exported = np.asarray(td["well_I1_perf_0_mass_rate_CO2"])
    np.testing.assert_allclose(exported, expected, rtol=1e-10)


def test_molar_and_mass_exports_are_consistent(thermal_run):
    """Component molar * molecular weight == component mass, row by row."""
    model, td = thermal_run
    mw_co2 = float(model.physics.property_containers[0].Mw[0])
    np.testing.assert_allclose(
        np.asarray(td["well_I1_perf_0_molar_rate_CO2"]) * mw_co2,
        np.asarray(td["well_I1_perf_0_mass_rate_CO2"]),
        rtol=1e-10,
    )


def test_perforation_sum_closes_the_material_balance(thermal_run):
    """Summed perforation rates equal the (single) perforation's, and at
    steady state they equal the wellhead mass-rate control target -- the
    material balance the zero export used to violate."""
    _, td = thermal_run
    np.testing.assert_array_equal(
        np.asarray(td["well_I1_mass_rate_CO2_by_sum_perfs"]),
        np.asarray(td["well_I1_perf_0_mass_rate_CO2"]),
    )
    final_sum = td["well_I1_mass_rate_G_by_sum_perfs"][-1]
    assert final_sum == pytest.approx(THERMAL_INJ_TARGET, rel=1e-6)


# --------------------------------------------- producer engine formulation


def test_producer_exports_the_molar_law_flux_with_production_sign(producer_run):
    """MOLAR-basis law on the BHP producer: the summed component molar rate is
    exactly ``PI * (p_well - p_res)`` -- negative, because the flow is from
    the reservoir into the well (the export convention is
    positive-for-injection, matching the Darcy branch)."""
    model, td = producer_run
    params = model.get_ipr_parameters()
    assert params["pi_type"] == PI_Type.MOLAR
    drawdown = _perforation_drawdown(model)
    expected = params["ipr_intercept"] + params["pi"] * (
        drawdown - params["ipr_pressure_offset"]
    )
    components = model.physics.property_containers[0].components_name
    exported = sum(
        np.asarray(td[f"well_I1_perf_0_molar_rate_{comp}"]) for comp in components
    )
    assert exported[-1] < 0.0, "a producing perforation must export a negative rate"
    np.testing.assert_allclose(exported, expected, rtol=1e-10)
    summed = sum(
        np.asarray(td[f"well_I1_molar_rate_{comp}_by_sum_perfs"]) for comp in components
    )
    np.testing.assert_allclose(summed, expected, rtol=1e-10)


# ------------------------------------------- non-molar (volumetric) basis


def test_volumetric_law_exports_the_total_volumetric_flux(volumetric_run):
    """VOLUMETRIC-basis law with nonzero intercept AND pressure offset: the
    exported phase volumetric rates sum to exactly
    ``A + B * (p_well - p_res - offset)``, per exported time."""
    model, td = volumetric_run
    params = model.get_ipr_parameters()
    assert params["pi_type"] == PI_Type.VOLUMETRIC
    assert params["ipr_intercept"] != 0.0 and params["ipr_pressure_offset"] != 0.0
    drawdown = _perforation_drawdown(model)
    expected = params["ipr_intercept"] + params["pi"] * (
        drawdown - params["ipr_pressure_offset"]
    )
    exported = np.asarray(td["well_I1_perf_0_volumetric_rate_G"]) + np.asarray(
        td["well_I1_perf_0_volumetric_rate_L"]
    )
    assert exported[-1] > 0.0
    np.testing.assert_allclose(exported, expected, rtol=1e-10)


# ---------------------- the C++ (legacy time_data) vs Python reporting pin


@pytest.fixture(scope="module")
def epm_pin_run(tmp_path_factory):
    """The thermal model plus a SECOND, EPM-type BHP producer whose single
    perforation carries a native MASS-basis IPR.

    EPM wells are the wells the legacy C++ reporting path serves:
    ``engine_base::post_newtonloop`` calls ``ms_well::calc_rates`` for them
    after every converged timestep, filling ``engine.time_data``. This is the
    configuration that pins the C++ law-aware reporting against the Python
    export mirror.
    """
    cls = _load_model_class(THERMAL_DIR, "native_ipr_rates_epm_pin_model")

    class _PinModel(cls):
        def set_wells(self):
            super().set_wells()
            self.reservoir.add_well("P2", ms_well.MS_Type.EPM, well_diameter=0.1)
            self.reservoir.add_perforation(
                "P2",
                res_cell_idx=(2, 1, 1),
                well_diameter=0.1,
                well_index=0.0,
                well_indexD=0.0,
                flow_law=LinearIPR(productivity=2.0e3, basis=PI_Type.MASS),
            )

        def set_well_controls(self):
            super().set_well_controls()
            well = self.reservoir.get_well("P2")
            self.physics.set_well_controls(
                wctrl=well.control,
                control_type=well_control_iface.BHP,
                is_inj=False,
                target=5.5,  # below the 5.88812 bar initial reservoir pressure
            )

        def set_initial_conditions(self):
            # the base model's version, with the DFM-only init_state guarded so
            # the EPM twin does not look itself up in the Pipe dict
            input_distribution = {
                self.physics.vars[0]: 5.88812,
                "temperature": 321.90000,
            }
            self.physics.set_initial_conditions_from_array(
                mesh=self.reservoir.mesh, input_distribution=input_distribution
            )
            for well in self.reservoir.wells:
                if well.ms_type == ms_well.MS_Type.DFM:
                    well.init_state = value_vector(
                        self.wells[
                            well.name
                        ].initial_conditions.initial_conditions_vector
                    )

    model = _PinModel(formulation="ipr_engine")
    td = _run(model, 30.0, tmp_path_factory.mktemp("native_ipr_epm_pin"))
    return model, td


def test_cpp_and_python_reporting_agree_bit_for_bit(epm_pin_run):
    """The legacy C++ path (``ms_well::calc_rates`` -> ``engine.time_data``)
    and the Python export must report the SAME law flux: both resolve to
    ``perforation_law_rates()`` (directly, and by pinned mirror), so any
    disagreement is a drift between assembly and reporting."""
    model, td = epm_pin_run
    cpp = np.asarray(model.physics.engine.time_data["P2 : p 0 c 0 rate (Kmol/day)"])
    python = np.asarray(td["well_P2_perf_0_molar_rate_CO2"])
    assert cpp.size == python.size, "one row per converged timestep on both sides"
    assert np.all(cpp < 0.0), "the EPM producer's law flux must export negative"
    np.testing.assert_allclose(python, cpp, rtol=1e-12)


def test_the_dfm_well_of_the_pin_model_still_exports_the_law(epm_pin_run):
    """Two wells, two laws, one engine: the DFM injector's export is law-aware
    at the same time as the EPM producer's."""
    model, td = epm_pin_run
    drawdown = _perforation_drawdown(model, "I1")
    np.testing.assert_allclose(
        np.asarray(td["well_I1_perf_0_mass_rate_CO2"]),
        THERMAL_MASS_PI * drawdown,
        rtol=1e-10,
    )


# ---------------------- two operator regions: export must follow the block's
# ---------------------- own region, not region 0 (finding F1)


class _ScaledDensity:
    """Forwarding density evaluator that scales the base phase density.

    Scaling the MASS density scales the molar density (``dens_m = dens / Mw``)
    by the same factor while leaving the mixture molecular weight
    (``rho_mass / rho_m``) unchanged, so a VOLUMETRIC-basis law rate
    (``q_total * rho_m``) scales cleanly by the factor.
    """

    def __init__(self, base, factor: float):
        self.base = base
        self.factor = factor

    def evaluate(self, *args, **kwargs):
        return self.factor * self.base.evaluate(*args, **kwargs)


#: region-1 density multiplier of the two-region fixture; any value far from 1
#: makes region-0-vs-region-1 confusion a first-order error
TWO_REGION_RHO_SCALE = 2.0
#: volumetric productivity of the two-region producer's law [m3/day/bar]
TWO_REGION_VOL_PI = 5.0


@pytest.fixture(scope="module")
def two_region_pin_run(tmp_path_factory):
    """The thermal model with TWO operator regions plus an EPM BHP producer
    whose law-carrying perforation sits in region 1.

    Region 1 reuses every region-0 property evaluator except density, which is
    scaled by ``TWO_REGION_RHO_SCALE`` — so any reporting path that evaluates
    the upstream reservoir block with the region-0 interpolator instead of the
    block's own region-1 interpolator mis-reports the VOLUMETRIC-basis law
    rate by exactly that factor (finding F1). The engine assembles the block
    region-aware through ``mesh.op_num``, so the C++ ``time_data`` reporting
    is the region-aware truth the Python export is pinned against.
    """
    from darts.physics.base.property_container import PropertyContainer

    cls = _load_model_class(THERMAL_DIR, "native_ipr_rates_two_region_model")

    class _TwoRegionModel(cls):
        def set_physics(self):
            super().set_physics()
            pc0 = self.physics.property_containers[0]
            pc1 = PropertyContainer(
                pc0.phases_name,
                pc0.components_name,
                Mw=pc0.Mw,
                eps_z=self.zero / 10,
                temperature=None,
                rock_comp=0,
            )
            for shared in (
                "flash_ev",
                "enthalpy_ev",
                "viscosity_ev",
                "conductivity_ev",
                "rel_perm_ev",
                "IFT_ev",
            ):
                setattr(pc1, shared, getattr(pc0, shared))
            pc1.density_ev = {
                ph: _ScaledDensity(ev, TWO_REGION_RHO_SCALE)
                for ph, ev in pc0.density_ev.items()
            }
            # same output-property keys as region 0, bound to this container,
            # so the per-region property interpolators stay layout-identical
            pc1.output_props = {"temperature": lambda: pc1.temperature}
            for j, ph in enumerate(pc0.phases_name):
                pc1.output_props["s" + ph] = lambda jj=j: pc1.sat[jj]
                pc1.output_props["rho" + ph] = lambda jj=j: pc1.dens[jj]
                pc1.output_props["mu" + ph] = lambda jj=j: pc1.mu[jj]
                for i, comp in enumerate(pc0.components_name):
                    pc1.output_props[f"x{comp}_in_{ph}_mass"] = (
                        lambda jj=j, ii=i: pc1.x_mass[jj, ii]
                    )
            self.physics.add_property_region(pc1, region=1)

        def set_reservoir(self):
            super().set_reservoir()
            # cell (1,1,1) [the DFM well's block] stays region 0; cell (2,1,1)
            # [the EPM producer's block] becomes region 1
            self.reservoir.global_data["op_num"] = np.array([0, 1])

        def set_wells(self):
            super().set_wells()
            self.reservoir.add_well("P2", ms_well.MS_Type.EPM, well_diameter=0.1)
            self.reservoir.add_perforation(
                "P2",
                res_cell_idx=(2, 1, 1),
                well_diameter=0.1,
                well_index=0.0,
                well_indexD=0.0,
                flow_law=LinearIPR(
                    productivity=TWO_REGION_VOL_PI, basis=PI_Type.VOLUMETRIC
                ),
            )

        def set_well_controls(self):
            super().set_well_controls()
            well = self.reservoir.get_well("P2")
            self.physics.set_well_controls(
                wctrl=well.control,
                control_type=well_control_iface.BHP,
                is_inj=False,
                target=5.5,  # below the 5.88812 bar initial reservoir pressure
            )

        def set_initial_conditions(self):
            input_distribution = {
                self.physics.vars[0]: 5.88812,
                "temperature": 321.90000,
            }
            self.physics.set_initial_conditions_from_array(
                mesh=self.reservoir.mesh, input_distribution=input_distribution
            )
            for well in self.reservoir.wells:
                if well.ms_type == ms_well.MS_Type.DFM:
                    well.init_state = value_vector(
                        self.wells[
                            well.name
                        ].initial_conditions.initial_conditions_vector
                    )

    model = _TwoRegionModel(formulation="ipr_engine")
    td = _run(model, 30.0, tmp_path_factory.mktemp("native_ipr_two_region"))
    return model, td


def _interpolated_ops(physics, itor, states_2d: np.ndarray) -> np.ndarray:
    """One reservoir-operator row per state, from the given interpolator."""
    from darts.engines import index_vector

    n_ops = physics.reservoir_operators[physics.regions[0]].n_ops
    n_state = getattr(physics, "n_state", physics.n_vars)
    n = states_2d.shape[0]
    values = value_vector(np.zeros(n * n_ops))
    dvalues = value_vector(np.zeros(n * n_ops * n_state))
    itor.evaluate_with_derivatives(
        value_vector(np.ascontiguousarray(states_2d, dtype=float).ravel()),
        index_vector(np.arange(n).astype(np.int32)),
        values,
        dvalues,
    )
    return np.asarray(values).reshape(n, n_ops)


def _molar_density_from_ops(physics, ops: np.ndarray) -> np.ndarray:
    """Upstream mixture molar density exactly as the law reporting builds it:
    ``rho_m = sum_j s_j * sum_c FLUX_OP[j, c]``."""
    layout = physics.reservoir_operators[physics.regions[0]]
    pc = physics.property_containers[physics.regions[0]]
    sat = ops[:, layout.SAT_OP : layout.SAT_OP + pc.nph]
    f = np.stack(
        [
            np.sum(
                ops[
                    :,
                    layout.FLUX_OP + j * layout.ne : layout.FLUX_OP
                    + j * layout.ne
                    + pc.nc_fl,
                ],
                axis=1,
            )
            for j in range(pc.nph)
        ],
        axis=1,
    )
    return np.sum(sat * f, axis=1)


def test_two_region_export_uses_the_perforated_blocks_region(two_region_pin_run):
    """A VOLUMETRIC law whose upstream reservoir block lives in operator
    region 1 must be exported with region-1 properties (finding F1):

    * modern Python export == legacy C++ export bit-for-bit (both resolve to
      the region-aware assembled flux), and
    * the exported molar rate is the law arithmetic on the REGION-1 molar
      density — which this fixture separates from region 0 by a factor of
      ``TWO_REGION_RHO_SCALE``, so region-0 evaluation cannot pass.
    """
    from darts.tools.hdf5_tools import load_hdf5_to_dict

    model, td = two_region_pin_run
    physics = model.physics

    python = np.asarray(td["well_P2_perf_0_molar_rate_CO2"])
    cpp = np.asarray(physics.engine.time_data["P2 : p 0 c 0 rate (Kmol/day)"])
    assert cpp.size == python.size
    assert np.all(cpp < 0.0), "the BHP producer's law flux must export negative"
    np.testing.assert_allclose(python, cpp, rtol=1e-12)

    # reconstruct the law arithmetic from the exported states, once with the
    # region-1 interpolator (correct) and once with region 0 (the F1 bug)
    h5 = load_hdf5_to_dict(model.output.well_filepath)
    well = model.reservoir.get_well("P2")
    well_block = int(well.well_body_idx) + int(well.perforations[0][0])
    res_block = int(well.perforations[0][1])
    cell_id = h5["dynamic"]["cell_id"]
    iw = int(np.where(cell_id == well_block)[0][0])
    ir = int(np.where(cell_id == res_block)[0][0])
    X = h5["dynamic"]["X"]
    q_tot = TWO_REGION_VOL_PI * (X[:, iw, 0] - X[:, ir, 0])
    assert np.all(q_tot < 0.0), "reservoir upstream throughout the run"

    states_up = X[:, ir, :]
    rho_m_r1 = _molar_density_from_ops(
        physics, _interpolated_ops(physics, physics.acc_flux_itor[1], states_up)
    )
    rho_m_r0 = _molar_density_from_ops(
        physics, _interpolated_ops(physics, physics.acc_flux_itor[0], states_up)
    )
    np.testing.assert_allclose(
        rho_m_r1,
        TWO_REGION_RHO_SCALE * rho_m_r0,
        rtol=1e-10,
        err_msg="fixture must actually separate the regions",
    )

    np.testing.assert_allclose(python, q_tot * rho_m_r1, rtol=1e-10)


# ----------------- the retired dummy perforation's output contract (finding F4)


@pytest.fixture(scope="module")
def hook_run(tmp_path_factory):
    """The thermal model in its default PYTHON-HOOK IPR formulation, whose
    well-reservoir coupling is a condition interface with NO perforation."""
    cls = _load_model_class(THERMAL_DIR, "native_ipr_rates_hook_model")
    model = cls(formulation=None)
    td = _run(model, 30.0, tmp_path_factory.mktemp("native_ipr_hook"))
    return model, td


#: the DELIBERATE well-series schema of a hook-coupled (perforation-free) DFM
#: well of the thermal model: wellhead + summed-perforation rate families plus
#: BHP/BHT — and NO per-perforation columns. The retired zero-well-index dummy
#: perforation used to add ten ``well_I1_perf_0_*`` columns that were
#: identically 0.0 (the Darcy export is ``operators * WI * dp`` with
#: ``WI == 0``); the real coupling flux is served by
#: ``LinearDFMWellIPRHook.connection_rates()`` instead.
HOOK_WELL_SERIES_SCHEMA = {
    "time",
    "well_I1_BHP",
    "well_I1_BHT",
    "well_I1_advective_heat_rate_G_at_wh",
    "well_I1_advective_heat_rate_G_by_sum_perfs",
    "well_I1_advective_heat_rate_L_at_wh",
    "well_I1_advective_heat_rate_L_by_sum_perfs",
    "well_I1_mass_rate_CO2_at_wh",
    "well_I1_mass_rate_CO2_by_sum_perfs",
    "well_I1_mass_rate_G_at_wh",
    "well_I1_mass_rate_G_by_sum_perfs",
    "well_I1_mass_rate_L_at_wh",
    "well_I1_mass_rate_L_by_sum_perfs",
    "well_I1_molar_rate_CO2_at_wh",
    "well_I1_molar_rate_CO2_by_sum_perfs",
    "well_I1_molar_rate_G_at_wh",
    "well_I1_molar_rate_G_by_sum_perfs",
    "well_I1_molar_rate_L_at_wh",
    "well_I1_molar_rate_L_by_sum_perfs",
    "well_I1_volumetric_rate_G_at_wh",
    "well_I1_volumetric_rate_G_by_sum_perfs",
    "well_I1_volumetric_rate_L_at_wh",
    "well_I1_volumetric_rate_L_by_sum_perfs",
}


def test_hook_well_series_schema_is_the_deliberate_one(hook_run):
    """The perforation-free hook well exports exactly the named schema — a
    change in either direction (a resurrected dummy-perforation column, or a
    silently dropped family) must fail HERE by name, not only as a wholesale
    reference-pickle mismatch."""
    _, td = hook_run
    assert set(td.keys()) == HOOK_WELL_SERIES_SCHEMA


def test_connection_rates_is_the_hook_flux_diagnostic(hook_run):
    """``LinearDFMWellIPRHook.connection_rates()`` reports the actual coupling
    flux the retired always-zero ``well_I1_perf_0_*`` columns never carried:
    the law arithmetic on the current engine state, with the assembly's own
    upstream property resolution."""
    from darts.pipes.linear_dfm_well_ipr import LinearDFMWellIPRHook

    model, _ = hook_run
    hooks = [
        item
        for item in model.conditions.items
        if isinstance(item, LinearDFMWellIPRHook)
    ]
    assert len(hooks) == 1
    (entry,) = hooks[0].connection_rates()

    assert entry["well_name"] == "I1"
    n_vars = model.physics.n_vars
    X = np.asarray(model.physics.engine.X)
    drawdown = (
        X[entry["well_block_index"] * n_vars] - X[entry["res_block_index"] * n_vars]
    )
    assert entry["total_rate"] == pytest.approx(THERMAL_MASS_PI * drawdown, rel=1e-12)
    assert entry["total_rate"] > 0.0, (
        "the injector's law flux points into the reservoir"
    )
    assert np.sum(entry["component_molar_rates"]) > 0.0
    assert entry["energy_rate"] > 0.0, "thermal physics reports the energy rate"


def test_two_region_advective_heat_follows_the_blocks_region(two_region_pin_run):
    """The advective-heat export of the two-region law connection must be the
    law arithmetic on REGION-1 operator rows, including the dead-state rows.

    Known fixture limitation (2026-08-25 adversarial verification): the
    density-only region scaling cancels exactly in the mixture molar
    enthalpies (``h = rho_h / rho_m`` — both scale), so this pin verifies the
    heat branch's full wiring (region-aware ``m_rate``, phase split, and the
    per-slot dead-state evaluation) but cannot discriminate the region used
    for the enthalpy FACTORS themselves; an independent reconstruction
    confirmed those are region-aware too.
    """
    from darts.engines import index_vector
    from darts.tools.hdf5_tools import load_hdf5_to_dict

    model, td = two_region_pin_run
    physics = model.physics
    layout = physics.reservoir_operators[physics.regions[0]]
    pc = physics.property_containers[physics.regions[0]]
    nph, nc_fl, ne = pc.nph, pc.nc_fl, layout.ne

    h5 = load_hdf5_to_dict(model.output.well_filepath)
    well = model.reservoir.get_well("P2")
    well_block = int(well.well_body_idx) + int(well.perforations[0][0])
    res_block = int(well.perforations[0][1])
    cell_id = h5["dynamic"]["cell_id"]
    iw = int(np.where(cell_id == well_block)[0][0])
    ir = int(np.where(cell_id == res_block)[0][0])
    X = h5["dynamic"]["X"]
    q_tot = TWO_REGION_VOL_PI * (X[:, iw, 0] - X[:, ir, 0])

    names = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in h5["dynamic"]["variable_names"]
    ]
    states_up = X[:, ir, :].copy()
    states_dead = states_up.copy()
    states_dead[:, names.index("pressure")] = 1.01325
    t_idx = names.index("enthalpy")
    states_dead[:, t_idx] = 273.15 + 15
    # PH physics: convert (p_dead, T_dead) to enthalpy via thermal_var_itor,
    # exactly as calc_rates_at_conns does
    n_tv = physics.thermal_var_operator.n_ops
    n = states_dead.shape[0]
    vals = value_vector(np.zeros(n * n_tv))
    dvals = value_vector(np.zeros(n * n_tv * physics.n_vars))
    physics.thermal_var_itor.evaluate_with_derivatives(
        value_vector(np.ascontiguousarray(states_dead, dtype=float).ravel()),
        index_vector(np.arange(n).astype(np.int32)),
        vals,
        dvals,
    )
    states_dead[:, t_idx] = np.asarray(vals).reshape(n, n_tv)[:, 0]

    def mixture(ops):
        sat = ops[:, layout.SAT_OP : layout.SAT_OP + nph]
        flux = np.stack(
            [
                ops[:, layout.FLUX_OP + j * ne : layout.FLUX_OP + (j + 1) * ne]
                for j in range(nph)
            ],
            axis=1,
        )
        f = np.sum(flux[:, :, :nc_fl], axis=2)
        rho_m = np.sum(sat * f, axis=1)
        rho_h = np.sum(sat * flux[:, :, nc_fl], axis=1)
        return sat, f, rho_m, rho_h

    itor_r1 = physics.acc_flux_itor[1]
    sat, f, rho_m, rho_h = mixture(_interpolated_ops(physics, itor_r1, states_up))
    _, _, rho_m_d, rho_h_d = mixture(_interpolated_ops(physics, itor_r1, states_dead))
    m_rate = q_tot * rho_m
    nu = sat * f / rho_m[:, None]
    expected = (m_rate * (rho_h / rho_m - rho_h_d / rho_m_d))[:, None] * nu

    exported = np.stack(
        [
            np.asarray(td[f"well_P2_perf_0_advective_heat_rate_{ph}"])
            for ph in pc.phases_name
        ],
        axis=1,
    )
    np.testing.assert_allclose(exported, expected, rtol=1e-10)


# ------------- a declared-stencil hook must coexist with an ordinary EPM well


def test_hook_with_an_ordinary_epm_well_initializes_and_runs(tmp_path_factory):
    """Regression for the ``well.n_segments`` AttributeError (2026-08-25
    adversarial verification): ``_existing_couplings`` read a non-exposed
    attribute for non-DFM wells, so ANY model combining a declared-stencil
    condition (the Python IPR hook) with a plain EPM Darcy well crashed in
    ``model.init()``. The EPM segment-connection count now replicates
    ``conn_mesh::add_wells`` (max perforation segment + 1)."""
    cls = _load_model_class(THERMAL_DIR, "native_ipr_rates_hook_epm_model")

    class _HookPlusEpmModel(cls):
        def set_wells(self):
            super().set_wells()
            self.reservoir.add_well("P2", ms_well.MS_Type.EPM, well_diameter=0.1)
            self.reservoir.add_perforation(
                "P2", res_cell_idx=(2, 1, 1), well_diameter=0.1
            )

        def set_well_controls(self):
            super().set_well_controls()
            well = self.reservoir.get_well("P2")
            self.physics.set_well_controls(
                wctrl=well.control,
                control_type=well_control_iface.BHP,
                is_inj=False,
                target=5.5,
            )

        def set_initial_conditions(self):
            input_distribution = {
                self.physics.vars[0]: 5.88812,
                "temperature": 321.90000,
            }
            self.physics.set_initial_conditions_from_array(
                mesh=self.reservoir.mesh, input_distribution=input_distribution
            )
            for well in self.reservoir.wells:
                if well.ms_type == ms_well.MS_Type.DFM:
                    well.init_state = value_vector(
                        self.wells[
                            well.name
                        ].initial_conditions.initial_conditions_vector
                    )

    model = _HookPlusEpmModel(formulation=None)  # the DECLARED-STENCIL hook
    td = _run(model, 10.0, tmp_path_factory.mktemp("hook_plus_epm"))
    # the hook well couples through its condition interface, the EPM producer
    # through its ordinary Darcy perforation — both must be live
    assert np.asarray(td["well_I1_mass_rate_CO2_at_wh"])[-1] != 0.0
    assert np.asarray(td["well_P2_molar_rate_CO2_by_sum_perfs"])[-1] < 0.0
