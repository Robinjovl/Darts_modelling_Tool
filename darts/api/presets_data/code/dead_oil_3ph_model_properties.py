"""Bundled property-container for the 3ph_do preset.

Shipped copy of the ``ModelProperties`` class from
``open-darts-api/models/3ph_do/model.py``.  Referenced by the
preset-side ``plugin_registry`` block in
``darts/api/presets_data/physics/dead_oil/3ph_do.json``.

This is an immiscible 3-phase dead-oil container — flash is a trivial
"each component in its own phase" assignment (no actual K-value flash).

Keep this file in sync with the canonical source at
``models/3ph_do/model.py``; there is no automated drift check.
"""

from __future__ import annotations

import numpy as np

from darts.physics.super.property_container import PropertyContainer


class ModelProperties(PropertyContainer):
    """Immiscible 3-phase dead-oil property container.

    Bundled copy of ``models/3ph_do/model.py:ModelProperties``.
    """

    def __init__(
        self,
        phases_name,
        components_name,
        Mw,
        eps_z: float = 1e-11,
        rock_comp: float = 1e-6,
    ):
        """
        :param phases_name: ordered list of phase names (e.g. ``["gas", "oil", "wat"]``)
        :type phases_name: list[str]
        :param components_name: ordered list of component names (e.g. ``["g", "o", "w"]``)
        :type components_name: list[str]
        :param Mw: molecular weight of each component
        :type Mw: list[float] | np.ndarray
        :param eps_z: minimum allowed component fraction (OBL stability)
        :type eps_z: float
        :param rock_comp: rock compressibility [1/bar]
        :type rock_comp: float
        """
        super().__init__(
            phases_name=phases_name,
            components_name=components_name,
            Mw=Mw,
            eps_z=eps_z,
            rock_comp=rock_comp,
            temperature=1.0,
        )

    def run_flash(self, pressure, temperature, zc, evaluate_PT: bool = None):
        """Trivial 3-phase immiscible flash.

        Each component goes entirely into its own phase (the diagonal
        identity assignment).  No K-value calculation is performed —
        composition is simply the input ``zc``.

        :param pressure: pressure [bar] (unused)
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param zc: overall composition vector
        :type zc: list[float] | np.ndarray
        :param evaluate_PT: required by the PropertyContainer interface but
            not used here
        :type evaluate_PT: bool | None
        :return: phase index array ``[0, 1, 2]``
        :rtype: np.ndarray
        """
        ph = np.array([0, 1, 2], dtype=np.intp)
        self.temperature = temperature

        for i in range(self.nc):
            self.x[i][i] = 1
        self.nu = zc

        return ph
