"""Bundled property-container for the 2ph_do_thermal preset.

Shipped copy of the ``ModelProperties`` class from
``open-darts-api/models/2ph_do_thermal/model.py``.  Referenced by the
preset-side ``plugin_registry`` block in
``darts/api/presets_data/physics/dead_oil/2ph_do_thermal.json`` so the
preset works in wheel installs where ``models/`` is not shipped.

Differs from :mod:`dead_oil_model_properties` in two places:

* ``super().__init__`` is called positionally and with ``temperature=None``
  so the thermal flow path is active.
* The composition slice in :meth:`evaluate` uses ``vec_state_as_np[1:self.nc]``
  rather than ``[1:]`` to skip the temperature element in the state vector.

Keep this file in sync with the canonical source at
``models/2ph_do_thermal/model.py``; there is no automated drift check.
"""

from __future__ import annotations

import numpy as np

from darts.physics.super.property_container import PropertyContainer


class ModelProperties(PropertyContainer):
    """Dead-oil 2-phase thermal property container.

    Bundled copy of ``models/2ph_do_thermal/model.py:ModelProperties``.
    """

    def __init__(self, phases_name, components_name, eps_z: float = 1e-11):
        """
        :param phases_name: ordered list of phase names (e.g. ``["wat", "oil"]``)
        :type phases_name: list[str]
        :param components_name: ordered list of component names (e.g. ``["w", "o"]``)
        :type components_name: list[str]
        :param eps_z: minimum allowed component fraction (OBL stability)
        :type eps_z: float
        """
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(
            phases_name, components_name, Mw, eps_z=eps_z, temperature=None
        )

    def evaluate(self, state):
        """Evaluate state operators for element-based thermal physics.

        :param state: ``[pressure, comp_0, ..., comp_{N-1}, temperature]`` vector
        :type state: list[float] | np.ndarray
        :return: None — operator values stored on ``self``
        """
        vec_state_as_np = np.asarray(state)
        self.pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(
            vec_state_as_np[1 : self.nc],
            1 - np.sum(vec_state_as_np[1 : self.nc]),
        )

        self.clean_arrays()
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0, 1], dtype=np.intp)

        for j in self.ph:
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(self.pressure)
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()

        self.nu = zc
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return

    def evaluate_at_cond(self, pressure, zc):
        """Evaluate density at standard conditions (used for rates).

        :param pressure: pressure [bar]
        :type pressure: float
        :param zc: composition vector
        :type zc: list[float] | np.ndarray
        :return: ``(sat, dens_m)`` tuple of saturations and molar densities
        :rtype: tuple[np.ndarray, list[float]]
        """
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
