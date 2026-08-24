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
