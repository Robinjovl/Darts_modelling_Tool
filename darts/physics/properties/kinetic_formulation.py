class KineticVarFormulation:
    """
    Marker for what a kinetic (non-equilibrium) phase's raw zc entry/entries
    represent -- one instance per kinetic phase, held by
    PropertyContainer.kin_formulation, dispatched via isinstance() in
    PropertyContainer._setup_kinetic_and_phase_idxs().

    This is a separate question from ``is_mole_fraction`` on
    ``Flash.set_kinetic_phase()``, which is NOT simply "this phase uses
    MoleFractionKinetic" -- see that flag's docstring. In particular, a
    BulkVolumeFractionKinetic phase still needs ``is_mole_fraction=True``
    whenever the model's fluid primary variables were pre-scaled to leave room for
    it in a shared sum-to-1 budget (e.g. it's the state vector's closure
    component) -- true for every kinetic-component model in this repo today, even
    though the zc entry itself represents a volume fraction, not a mole fraction.
    """


class BulkVolumeFractionKinetic(KineticVarFormulation):
    """
    The raw zc entry already is the bulk volume fraction -- today's (implicit)
    behavior for every kinetic component in the codebase (see e.g. 2ph_comp_solid
    and Chem_benchmark_new, both of which document this raw entry as a saturation).
    Restricted to exactly one component per phase (see PropertyContainer).

    Whether to also pass Flash.set_kinetic_phase(..., is_mole_fraction=True) or
    ``False`` for this component is a separate question -- see that flag's
    docstring -- and depends on whether the fluid primary variables were
    pre-scaled to share a sum-to-1 budget with this entry, not on this class.
    Both current models need ``True`` there despite using this class.
    """


class MoleFractionKinetic(KineticVarFormulation):
    """
    The raw zc entry is a mole fraction of the combined (fluid + kinetic) total --
    the same simplex the fluid components live in, so it always needs
    Flash.set_kinetic_phase(..., is_mole_fraction=True) (the default). Such a phase
    is pooled directly into the fluid saturation/accumulation/diffusion/conduction
    machinery (see PropertyContainer.compute_saturation() and
    operator_evaluator.py), not given separate treatment -- unlike
    BulkVolumeFractionKinetic, it isn't restricted to one component per phase.
    """
