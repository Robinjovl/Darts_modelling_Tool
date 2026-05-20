"""Bundled property-container for dead-oil presets.

This is a shipped copy of the ``ModelProperties`` class from
``open-darts-api/models/2ph_do/model.py``.  It is referenced by the
preset-side ``plugin_registry`` block (see
``darts/api/presets_data/physics/dead_oil/cpg_deadoil_brugge.json``)
so the preset works in **wheel installs** where ``models/`` is not
shipped.

Keep this file in sync with the canonical source at
``models/2ph_do/model.py`` — there is no automated check.
"""

from __future__ import annotations

import numpy as np

from darts.physics.super.property_container import PropertyContainer


class ModelProperties(PropertyContainer):
    """Dead-oil 2-phase property container.

    Bundled copy of ``models/2ph_do/model.py:ModelProperties`` so
    preset-side ``plugin_registry`` blocks resolve at wheel-install
    time.
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
            phases_name=phases_name,
            components_name=components_name,
            Mw=Mw,
            eps_z=eps_z,
            temperature=1.0,
        )

    def evaluate(self, state):
        """Evaluate state operators for element-based physics.

        :param state: ``[pressure, comp_0, ..., comp_{N-1}]`` state vector
        :type state: list[float] | np.ndarray
        :return: None — operator values stored on ``self``
        """
        vec_state_as_np = np.asarray(state)
        self.pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

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
