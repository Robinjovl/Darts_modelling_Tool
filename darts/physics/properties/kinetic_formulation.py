import abc

import numpy as np


class KineticVarFormulation(abc.ABC):
    """
    Defines what a kinetic (non-equilibrium) component's raw zc entry actually
    represents, and how to convert it to the quantities the rest of PropertyContainer
    and the operators need. One instance per kinetic component, held by
    PropertyContainer.kin_formulation.

    This is a separate question from ``is_mole_fraction`` on
    ``Flash.set_kinetic_phase()``, which is NOT simply "this component uses
    MoleFractionKinetic" -- see that flag's docstring. In particular, a
    BulkVolumeFractionKinetic component still needs ``is_mole_fraction=True``
    whenever the model's fluid primary variables were pre-scaled to leave room for
    it in a shared sum-to-1 budget (e.g. it's the state vector's closure
    component) -- true for every kinetic-component model in this repo today, even
    though the zc entry itself represents a volume fraction, not a mole fraction.
    """

    @abc.abstractmethod
    def to_bulk_volume_fraction(
        self,
        raw_value: float,
        dens_m_kin: float,
        nu: np.ndarray,
        dens_m: np.ndarray,
    ) -> float:
        """
        Convert the raw zc entry for this kinetic component to a bulk volume
        fraction (V_kin / V_bulk) -- what ACC_OP/UPSAT_OP/porosity accounting need.

        :param raw_value: The raw zc entry for this kinetic component.
        :param dens_m_kin: This kinetic phase's molar density [kmol/m3].
        :param nu: Molar fraction of the combined (fluid + kinetic) total, per phase
                   (PropertyContainer.nu, length nph) -- only phases with nu > 0 are
                   actually present. Formulations that don't need cross-phase context
                   (e.g. BulkVolumeFractionKinetic) can ignore this.
        :param dens_m: Molar density [kmol/m3] per phase (PropertyContainer.dens_m,
                       length nph), aligned with ``nu``.
        """


class BulkVolumeFractionKinetic(KineticVarFormulation):
    """
    The raw zc entry already is the bulk volume fraction -- today's (implicit)
    behavior for every kinetic component in the codebase (see e.g. 2ph_comp_solid
    and Chem_benchmark_new, both of which document this raw entry as a saturation).

    Whether to also pass Flash.set_kinetic_phase(..., is_mole_fraction=True) or
    ``False`` for this component is a separate question -- see that flag's
    docstring -- and depends on whether the fluid primary variables were
    pre-scaled to share a sum-to-1 budget with this entry, not on this class.
    Both current models need ``True`` there despite using this class.
    """

    def to_bulk_volume_fraction(self, raw_value, dens_m_kin, nu, dens_m):
        return raw_value


class MoleFractionKinetic(KineticVarFormulation):
    """
    The raw zc entry is a mole fraction of the combined (fluid + kinetic) total --
    the same simplex the fluid components live in, so it always needs
    Flash.set_kinetic_phase(..., is_mole_fraction=True) (the default): Flash then
    normalizes the equilibrium feed and this phase's own nu on that same total-moles
    basis, so ``raw_value`` (and PropertyContainer.nu for this phase, which Flash
    sets to the same value) already is n_kin / n_total. (Unlike
    BulkVolumeFractionKinetic, this is a hard requirement, not a model-specific
    detail -- a mole-fraction entry is by definition part of that simplex.)

    Converting that to a bulk volume fraction mirrors
    PropertyContainer.compute_saturation(): this phase's volume share (n/dens_m) of
    the combined volume of every phase present (fluid and kinetic alike), moles
    canceling out of the ratio so n_total is never actually needed.
    """

    def to_bulk_volume_fraction(self, raw_value, dens_m_kin, nu, dens_m):
        present = nu > 0
        vol = nu[present] / dens_m[present]
        vol_kin = raw_value / dens_m_kin
        return vol_kin / np.sum(vol)
