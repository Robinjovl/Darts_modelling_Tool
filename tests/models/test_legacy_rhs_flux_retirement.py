"""M4 wave 2: the legacy Python-side RHS channels are gone from ``DartsModel``.

``set_rhs_flux()`` (an override the model returned a dense vector from) and
``rhs_flux_hooks`` (an untyped list of objects with an ``apply(dt, t)`` method)
were the two ad-hoc channels the unified conditions layer replaced. This module
pins what wave 2 changed:

* ``DartsModel`` no longer DEFINES ``set_rhs_flux``, and no longer OWNS a
  ``rhs_flux_hooks`` list -- ``apply_rhs_flux`` is the conditions stage plus the
  observer stage and nothing else;
* the two spellings that out-of-tree models still use are adapters ONTO that
  layer: ``rhs_flux_hooks.append(item)`` registers on ``model.conditions``, and
  an inherited ``set_rhs_flux`` override is applied by a
  :class:`LegacyRhsFluxOverride` item registered FIRST (where the legacy stage
  ran), with the same ``rhs += value * dt`` arithmetic;
* both shipped ``darts.pipes`` hooks are ``ConditionItem`` subclasses, so the
  registration alias hands them to the typed contract rather than to a list.
"""

import numpy as np
import pytest

pytest.importorskip("darts.engines")

from darts.models.conditions import (  # noqa: E402
    AssemblyContext,
    CellSource,
    ConditionItem,
    ConditionSet,
    LegacyHookRegistration,
    LegacyRhsFluxOverride,
)
from darts.models.darts_model import DartsModel  # noqa: E402

N_VARS = 2


def _bare(cls):
    """A DartsModel subclass instance with only the attributes under test."""
    model = object.__new__(cls)
    model.conditions = ConditionSet()
    return model


# ------------------------------------------------------- the base class surface
def test_darts_model_no_longer_defines_set_rhs_flux():
    assert not hasattr(DartsModel, "set_rhs_flux")


def test_rhs_flux_hooks_is_not_a_list_but_a_registration_alias():
    model = _bare(DartsModel)
    hooks = model.rhs_flux_hooks
    assert isinstance(hooks, LegacyHookRegistration)
    assert not isinstance(hooks, list)


def test_rhs_flux_hooks_cannot_be_assigned():
    """It is a read-only property: nothing may install a second registry."""
    model = _bare(DartsModel)
    with pytest.raises(AttributeError):
        model.rhs_flux_hooks = []


# ------------------------------------------------- the hook-list registration alias
def test_append_registers_on_conditions_and_warns():
    model = _bare(DartsModel)
    item = CellSource(cells=[0], rates=[[1.0, 1.0]])
    with pytest.deprecated_call(match="rhs_flux_hooks"):
        returned = model.rhs_flux_hooks.append(item)
    assert returned is item
    assert model.conditions.items == [item]
    # the alias reads the item list back
    assert len(model.rhs_flux_hooks) == 1
    assert list(model.rhs_flux_hooks) == [item]
    assert item in model.rhs_flux_hooks
    assert model.rhs_flux_hooks[0] is item


def test_append_of_a_non_condition_item_is_refused_by_name():
    class _NotAnItem:
        def apply(self, dt, t):
            pass

    model = _bare(DartsModel)
    with pytest.raises(TypeError, match="ConditionItem"):
        with pytest.warns(DeprecationWarning):
            model.rhs_flux_hooks.append(_NotAnItem())
    assert model.conditions.items == []


def test_extend_registers_every_item():
    model = _bare(DartsModel)
    items = [
        CellSource(cells=[0], rates=[[1.0, 1.0]]),
        CellSource(cells=[1], rates=[[2.0, 2.0]]),
    ]
    with pytest.warns(DeprecationWarning):
        model.rhs_flux_hooks.extend(items)
    assert model.conditions.items == items


# --------------------------------------------- the set_rhs_flux override adapter
class _WithOverride(DartsModel):
    def __init__(self):
        # deliberately NOT calling DartsModel.__init__ (it builds a whole model);
        # the registration path is exercised explicitly below.
        self.conditions = ConditionSet()

    def set_rhs_flux(self, t=None):
        return np.array([1.0, 2.0, 3.0, 4.0])


class _WithoutOverride(DartsModel):
    def __init__(self):
        self.conditions = ConditionSet()


def test_a_model_without_an_override_registers_nothing():
    model = _WithoutOverride()
    model._register_legacy_rhs_flux_override()
    assert model.conditions.items == []


def test_an_override_is_registered_first_and_warns():
    model = _WithOverride()
    model.conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    with pytest.deprecated_call(match="set_rhs_flux"):
        model._register_legacy_rhs_flux_override()
    # registered LAST here because the source was added by hand before it; the
    # real ordering guarantee comes from DartsModel.__init__, which runs the
    # registration before set_wells() can add anything. What matters for the
    # contract is that exactly one adapter is registered, bound to this model.
    (adapter,) = [
        item for item in model.conditions if isinstance(item, LegacyRhsFluxOverride)
    ]
    assert adapter.model is model


def test_the_adapter_applies_value_times_dt_with_the_legacy_sign():
    model = _WithOverride()
    adapter = LegacyRhsFluxOverride(model)
    rhs = np.zeros(4)
    ctx = AssemblyContext(
        rhs=rhs,
        jac=None,
        X=np.zeros(4),
        Xn=np.zeros(4),
        dt=0.25,
        t=3.0,
        iteration=0,
        n_vars=N_VARS,
        n_res_blocks=2,
    )
    adapter.apply(ctx)
    np.testing.assert_array_equal(rhs, np.array([1.0, 2.0, 3.0, 4.0]) * 0.25)


def test_the_adapter_is_an_additive_item_claiming_no_row():
    adapter = LegacyRhsFluxOverride(_WithOverride())
    assert isinstance(adapter, ConditionItem)
    assert adapter.contribution == "additive"
    assert adapter.provides_jacobian is False
    assert list(adapter.written_rows(None)) == []


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


def test_the_shipped_models_register_through_conditions_not_a_hook_list():
    """A grep-proof: no model in the repository uses the legacy spelling."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    offenders = [
        path
        for path in (root / "models").rglob("*.py")
        if "rhs_flux_hooks" in path.read_text()
    ]
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

    class _Mesh:
        n_res_blocks = 2

    class _Reservoir:
        mesh = _Mesh()

    class _Model(DartsModel):
        def __init__(self):
            self.conditions = ConditionSet()
            self.physics = _Physics()
            self.physics.engine = _Engine()
            self.reservoir = _Reservoir()
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
