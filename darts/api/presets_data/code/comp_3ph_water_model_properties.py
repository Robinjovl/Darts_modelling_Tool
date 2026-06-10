"""Bundled property-container for the 3ph_comp_w preset.

Shipped copy of the ``ModelProperties`` class from
``open-darts-api/models/3ph_comp_w/model.py``.  Referenced by the
preset-side ``plugin_registry`` block in
``darts/api/presets_data/physics/compositional/3ph_comp_w.json``.

The container runs a real K-value flash on the non-water components and
treats water as an immiscible third phase that always sits in its own
slot (last component, last phase).

Keep this file in sync with the canonical source at
``models/3ph_comp_w/model.py``; there is no automated drift check.
"""

from __future__ import annotations

import numpy as np

from darts.physics.super.property_container import PropertyContainer


class ModelProperties(PropertyContainer):
    """3-phase compositional with immiscible water phase.

    Bundled copy of ``models/3ph_comp_w/model.py:ModelProperties``.
    """

    def __init__(self, phases_name, components_name, Mw, eps_z: float = 1e-11):
        """
        :param phases_name: ordered phase names (e.g. ``["gas", "oil", "wat"]``)
        :type phases_name: list[str]
        :param components_name: ordered component names (water last)
        :type components_name: list[str]
        :param Mw: molecular weights
        :type Mw: list[float] | np.ndarray
        :param eps_z: minimum allowed component fraction (OBL stability)
        :type eps_z: float
        """
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=1.0)

    def run_flash(self, pressure, temperature, zc, evaluate_PT: bool = None):
        """K-value flash on non-water components; water as immiscible third phase.

        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param zc: overall composition vector (water last)
        :type zc: list[float] | np.ndarray
        :param evaluate_PT: required by the PropertyContainer interface but
            not used here
        :type evaluate_PT: bool | None
        :return: phase index array
        :rtype: np.ndarray
        """
        zc_r = zc[:-1] / (1 - zc[-1])
        self.flash_ev.evaluate(pressure, temperature, zc_r)
        self.temperature = temperature
        flash_results = self.flash_ev.get_flash_results()
        nu = np.array(flash_results.nu)
        xr = np.array(flash_results.X).reshape(self.nph - 1, self.nc - 1)
        V = nu[0]

        if V <= 0:
            V = 0
            xr[1] = zc_r
            ph = [1, 2]
        elif V >= 1:
            V = 1
            xr[0] = zc_r
            ph = [0, 2]
        else:
            ph = [0, 1, 2]

        for i in range(self.nc - 1):
            for j in range(2):
                self.x[j][i] = xr[j][i]

        self.x[-1][-1] = 1

        self.nu[0] = V * (1 - zc[-1])
        self.nu[1] = (1 - V) * (1 - zc[-1])
        self.nu[2] = zc[-1]

        return np.array(ph, dtype=np.intp)
