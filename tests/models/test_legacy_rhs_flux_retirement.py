"""The legacy Python-side RHS channels are GONE -- not deprecated, removed.

``set_rhs_flux()`` (an override the model returned a dense
``n_blocks * n_vars`` vector from) and ``rhs_flux_hooks`` (an untyped list of
objects with an ``apply(dt, t)`` method) were the two ad-hoc channels the
unified conditions layer replaced. They were kept for one wave as adapters onto
that layer; this module pins their removal:

* ``DartsModel`` defines neither name, and ``darts.models.conditions`` no longer
  ships the ``LegacyRhsFluxOverride`` / ``LegacyHookRegistration`` adapters --
  ``apply_rhs_flux`` is the conditions stage plus the observer stage and nothing
  else;
* removing a channel must not turn a working model into a quietly wrong one, so
  the two spellings that would otherwise fail SILENTLY are refused at
  ``init()`` time by :meth:`ConditionSet.compile`, with the migration named:
  an inherited ``set_rhs_flux`` override (nothing would call it) and a
  self-assigned ``rhs_flux_hooks`` list (nothing would read it).
  ``self.rhs_flux_hooks.append(...)`` needs no such check -- there is no such
  attribute, so it raises by itself;
* nothing under ``darts/``, ``models/`` or ``tutorials/`` spells either name any
  more, and both shipped ``darts.pipes`` hooks are ``ConditionItem`` subclasses.
"""

import pathlib

import numpy as np
import pytest

pytest.importorskip("darts.engines")

import darts.models.conditions as conditions_module  # noqa: E402
from darts.models.conditions import (  # noqa: E402
    AssemblyContext,
    CellSource,
    ConditionItem,
    ConditionSet,
    PipeSourceTerm,
)
from darts.models.darts_model import DartsModel  # noqa: E402

N_VARS = 2


class _StubMesh:
    n_blocks = 2
    n_res_blocks = 2


class _StubReservoir:
    mesh = _StubMesh()


def _bare(cls):
    """A DartsModel subclass instance with only the attributes under test."""
    model = object.__new__(cls)
    model.conditions = ConditionSet()
    model.reservoir = _StubReservoir()
    return model


# ------------------------------------------------------- the base class surface
def test_darts_model_defines_neither_legacy_name():
    assert not hasattr(DartsModel, "set_rhs_flux")
    assert not hasattr(DartsModel, "rhs_flux_hooks")


def test_the_adapters_are_gone_from_the_conditions_module():
    assert not hasattr(conditions_module, "LegacyRhsFluxOverride")
    assert not hasattr(conditions_module, "LegacyHookRegistration")


def test_appending_to_rhs_flux_hooks_raises_instead_of_registering():
    """The spelling fails loudly by itself: there is nothing to append to."""
    model = _bare(DartsModel)
    with pytest.raises(AttributeError, match="rhs_flux_hooks"):
        model.rhs_flux_hooks.append(CellSource(cells=[0], rates=[[1.0, 1.0]]))


# ------------------------------ the two spellings that would fail SILENTLY
class _WithOverride(DartsModel):
    def set_rhs_flux(self, t=None):
        return np.array([1.0, 2.0, 3.0, 4.0])


class _WithOwnHookList(DartsModel):
    pass


class _Plain(DartsModel):
    pass


def test_compile_refuses_a_surviving_set_rhs_flux_override():
    model = _bare(_WithOverride)
    with pytest.raises(RuntimeError, match="set_rhs_flux"):
        model.conditions.compile(model)


def test_compile_refuses_it_even_when_nothing_is_registered():
    """The dangerous model is exactly the one that registers nothing."""
    model = _bare(_WithOverride)
    assert model.conditions.items == []
    with pytest.raises(RuntimeError, match="conditions"):
        model.conditions.compile(model)


def test_the_refusal_names_the_sign_flip():
    model = _bare(_WithOverride)
    with pytest.raises(RuntimeError, match="positive INTO the cell"):
        model.conditions.compile(model)


def test_compile_refuses_a_self_assigned_rhs_flux_hooks_list():
    model = _bare(_WithOwnHookList)
    model.rhs_flux_hooks = []
    model.rhs_flux_hooks.append(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    with pytest.raises(RuntimeError, match="rhs_flux_hooks"):
        model.conditions.compile(model)


def test_a_migrated_model_compiles():
    model = _bare(_Plain)
    assert model.conditions.compile(model) is model.conditions


# ------------------- the item the DFM-well overrides were replaced BY
#
# The removal only holds if PipeSourceTerm does what those set_rhs_flux()
# overrides did. They all had the same body: read the source/sink's current
# component+energy rates at the specific potential energy of the RECEIVING
# block, write them NEGATED into a dense n_blocks * n_vars vector at
# (n_res_blocks + segment_idx) * n_vars, and let the caller apply
# ``rhs += vector * dt``. That composite is reproduced here, literally, and
# compared against the item.


class _StubSourceSink:
    """The get_component_energy_rates() surface of a pipe source/sink."""

    def __init__(self, segment_idx, rate, composition, molar_enthalpy, mw_avg):
        self.segment_idx = segment_idx
        self.rate = rate
        self.composition = np.asarray(composition, dtype=float)
        self.molar_enthalpy = molar_enthalpy
        self.mw_avg = mw_avg

    def get_component_energy_rates(self, physics, specific_potential_energy=0.0):
        component_rate = self.rate * self.composition
        if not physics.thermal:
            return component_rate
        molar_energy = self.molar_enthalpy + specific_potential_energy * self.mw_avg
        return np.append(component_rate, self.rate * molar_energy)


class _StubPipe:
    def __init__(self, source_sinks):
        self.source_sinks = source_sinks


class _PipeMesh:
    def __init__(self, n_res_blocks, n_blocks, cell_spe):
        self.n_res_blocks = n_res_blocks
        self.n_blocks = n_blocks
        self.cell_spe = np.asarray(cell_spe, dtype=float)


class _PipePhysics:
    def __init__(self, n_vars, thermal):
        self.n_vars = n_vars
        self.thermal = thermal


def _pipe_model(thermal, n_vars, segment_idx=0):
    n_res_blocks, n_blocks = 2, 5
    source = _StubSourceSink(
        segment_idx=segment_idx,
        rate=58895.98,
        composition=np.ones(n_vars - 1 if thermal else n_vars),
        molar_enthalpy=-21234.5,
        mw_avg=44.01,
    )

    class _Model:
        pass

    model = _Model()
    model.wells = {"I1": _StubPipe({"RampUpRate1": source})}
    model.physics = _PipePhysics(n_vars, thermal)

    class _Res:
        pass

    model.reservoir = _Res()
    model.reservoir.mesh = _PipeMesh(
        n_res_blocks, n_blocks, np.arange(n_blocks) * -3.75
    )
    return model, source


def _legacy_override_result(model, source, dt):
    """The body every migrated set_rhs_flux() override had, verbatim."""
    mesh, physics = model.reservoir.mesh, model.physics
    specific_potential_energy = mesh.cell_spe[mesh.n_res_blocks + source.segment_idx]
    rates = source.get_component_energy_rates(physics, specific_potential_energy)
    rhs_flux = np.zeros(mesh.n_blocks * physics.n_vars)
    start = (mesh.n_res_blocks + source.segment_idx) * physics.n_vars
    rhs_flux[start : start + physics.n_vars] = -rates
    return rhs_flux * dt  # the caller's ``rhs += rhs_flux * dt``


def _apply(item, model, dt):
    rhs = np.zeros(model.reservoir.mesh.n_blocks * model.physics.n_vars)
    item.bind(model)
    item.apply(
        AssemblyContext(
            rhs=rhs,
            jac=None,
            X=np.zeros_like(rhs),
            Xn=np.zeros_like(rhs),
            dt=dt,
            t=0.0,
            iteration=0,
            n_vars=model.physics.n_vars,
            n_res_blocks=model.reservoir.mesh.n_res_blocks,
        )
    )
    return rhs


@pytest.mark.parametrize("thermal, n_vars", [(True, 3), (False, 2)])
def test_pipe_source_term_reproduces_the_legacy_override_exactly(thermal, n_vars):
    dt = 2.31481e-05
    model, source = _pipe_model(thermal, n_vars)
    item = PipeSourceTerm(well_name="I1", source_sink_name="RampUpRate1")
    np.testing.assert_array_equal(
        _apply(item, model, dt), _legacy_override_result(model, source, dt)
    )


def test_pipe_source_term_writes_only_the_receiving_block():
    model, source = _pipe_model(thermal=True, n_vars=3, segment_idx=1)
    item = PipeSourceTerm(well_name="I1", source_sink_name="RampUpRate1")
    rhs = _apply(item, model, dt=1.0)
    block = model.reservoir.mesh.n_res_blocks + source.segment_idx
    touched = np.flatnonzero(rhs) // model.physics.n_vars
    assert set(touched.tolist()) == {block}
    assert list(item.written_rows(model)) == [
        block * model.physics.n_vars + equation for equation in range(3)
    ]


# ------------------------------------------------ the two shipped pipes hooks
def test_both_pipes_hooks_are_condition_items():
    from darts.pipes.add_lateral_heat_exchange import (
        SemiAnalyticalWellLateralHeatTransferHook,
    )
    from darts.pipes.linear_dfm_well_ipr import LinearDFMWellIPRHook

    assert issubclass(LinearDFMWellIPRHook, ConditionItem)
    assert issubclass(SemiAnalyticalWellLateralHeatTransferHook, ConditionItem)
    # the IPR item writes four dense blocks, so it declares the Jacobian up front
    assert LinearDFMWellIPRHook.provides_jacobian is True


# --------------------------------------------------------------- the grep-proof
def test_no_python_source_spells_either_legacy_name():
    root = pathlib.Path(__file__).resolve().parents[2]
    # conditions.py is the one module allowed to name them: it is where the
    # refusal above lives, and a refusal has to say what it refuses.
    allowed = {pathlib.Path("darts/models/conditions.py")}
    offenders = []
    for tree in ("darts", "models", "tutorials"):
        for path in (root / tree).rglob("*.py"):
            relative = path.relative_to(root)
            if relative in allowed:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "def set_rhs_flux" in text or ".rhs_flux_hooks" in text:
                offenders.append(str(relative))
    assert offenders == []


# ------------------------------------------- apply_rhs_flux has no legacy stage
def test_apply_rhs_flux_runs_conditions_then_observers_then_after_assembly():
    order = []

    class _Item(ConditionItem):
        def apply(self, ctx):
            order.append("item")

    class _Engine:
        def __init__(self):
            self.RHS = np.zeros(4)
            self.X = np.zeros(4)
            self.Xn = np.zeros(4)
            self.jac_diags = np.array([], dtype=np.int64)

    class _Physics:
        n_vars = N_VARS

    class _Model(DartsModel):
        def __init__(self):
            self.conditions = ConditionSet()
            self.physics = _Physics()
            self.physics.engine = _Engine()
            self.reservoir = _StubReservoir()
            self.nonlinear_solver = None
            self._conditions_csr_view = None
            self._assembly_iteration = 0
            self._pattern_version = 0

        def after_assembly(self, dt, t):
            order.append("after_assembly")

    model = _Model()
    model.conditions.add(_Item())
    model.apply_rhs_flux(0.5, 1.0)
    assert order == ["item", "after_assembly"]
