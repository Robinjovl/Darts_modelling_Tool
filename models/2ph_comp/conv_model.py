"""Convergence-study model for the WENO2 transport scheme review.

Two regimes on a FIXED physical domain [0, L] (so dx = L/nx varies with nx):

  * 'smooth'  : a uniform two-phase background with a smooth cosine bump in the
                CO2 overall composition, advected by a uniform Darcy flow driven
                by a constant pressure drop.  Mobility is made ~uniform (equal
                phase viscosities, straight-line relative permeability) so the
                bump rides on a near-constant velocity and stays smooth -> this
                is the regime where WENO2 should show ~2nd-order spatial
                convergence and SPU ~1st order (paper Eq. 11-14; note the paper
                requires dt ~ dx^2 to *observe* 2nd order with backward Euler).

  * 'sharp'   : a step in CO2 composition (Riemann-like) advected the same way;
                the self-sharpening front tests oscillation-free / bounded
                behaviour and the reduced (~1/2..1) convergence order expected
                at a discontinuity.

The transported diagnostic is the CO2 overall composition z_CO2 = X[1::nc].
"""

import numpy as np

from darts.engines import sim_params, well_control_iface
from darts.models.cicd_model import CICDModel
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.flash import ConstantK
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer


class ConvModel(CICDModel):
    def __init__(
        self,
        transport_scheme: str = "spu",
        nx: int = 100,
        domain_length: float = 100.0,
        regime: str = "smooth",
        n_steps: int = 100,
        runtime: float = 200.0,
        bump_amp: float = 0.25,
        z_bg=(0.5, 0.2),          # (z_CO2, z_C1) background; z_H2O = rest
        uniform_mobility: bool = True,
    ):
        super().__init__()
        self.timer.node["initialization"].start()

        self.nx = nx
        self.domain_length = domain_length
        self.regime = regime
        self.bump_amp = bump_amp
        self.z_bg = z_bg
        self.uniform_mobility = uniform_mobility
        self.runtime = runtime

        self.set_reservoir(nx, domain_length)
        self.set_physics(uniform_mobility)

        # dt ~ dx^2 is imposed by the driver through n_steps; here we just set a
        # uniform max timestep = runtime / n_steps and disable growth so every
        # step is the same size (clean temporal error control).
        dt = runtime / n_steps
        self.set_sim_params(
            first_ts=dt,
            mult_ts=1.0,
            max_ts=dt,
            runtime=runtime,
            tol_newton=1e-4,
            tol_linear=1e-6,
            it_newton=20,
            it_linear=100,
            newton_type=sim_params.newton_local_chop,
        )
        schemes = {"spu": sim_params.spu, "weno2": sim_params.weno2}
        self.params.transport_scheme = schemes[transport_scheme.lower()]
        self.transport_scheme = transport_scheme.lower()

        self.timer.node["initialization"].stop()

    def set_reservoir(self, nx, L):
        dx = L / nx
        self.reservoir = StructReservoir = __import__(
            "darts.reservoirs.struct_reservoir", fromlist=["StructReservoir"]
        ).StructReservoir(
            self.timer,
            nx=nx, ny=1, nz=1,
            dx=dx, dy=10.0, dz=10.0,
            permx=100.0, permy=100.0, permz=100.0,
            poro=0.3,
            depth=1000.0,
        )

    def set_physics(self, uniform_mobility):
        zero = 1e-8
        epsilon = 1e-9
        components = ['CO2', 'C1', 'H2O']
        phases = ['gas', 'aqueous']
        Mw = [44.01, 16.04, 18.015]

        pc = PropertyContainer(
            phases_name=phases, components_name=components, Mw=Mw,
            eps_z=epsilon, temperature=1.0,
        )
        pc.flash_ev = ConstantK(len(components), [4, 2, 1e-1], zero)

        if uniform_mobility:
            # Equal, ~incompressible densities + equal viscosities + straight-line
            # relperm -> total mobility nearly constant -> uniform advection speed.
            pc.density_ev = dict([
                ('gas', DensityBasic(compr=1e-6, dens0=500)),
                ('aqueous', DensityBasic(compr=1e-6, dens0=500)),
            ])
            pc.viscosity_ev = dict([('gas', ConstFunc(0.1)), ('aqueous', ConstFunc(0.1))])
            pc.rel_perm_ev = dict([
                ('gas', PhaseRelPerm("gas", swc=0.0, sgr=0.0, n=1.0)),
                ('aqueous', PhaseRelPerm("oil", swc=0.0, sgr=0.0, n=1.0)),
            ])
        else:
            pc.density_ev = dict([
                ('gas', DensityBasic(compr=1e-3, dens0=200)),
                ('aqueous', DensityBasic(compr=1e-5, dens0=600)),
            ])
            pc.viscosity_ev = dict([('gas', ConstFunc(0.05)), ('aqueous', ConstFunc(0.5))])
            pc.rel_perm_ev = dict([
                ('gas', PhaseRelPerm("gas")), ('aqueous', PhaseRelPerm("oil")),
            ])

        state_spec = Compositional.StateSpecification.P
        p_step = (300 - 1) / (200 - 1)
        z_step = (1 - 3 * epsilon) / (200 - 1)
        self.physics = Compositional(
            components, phases, self.timer,
            state_spec=state_spec,
            axes_step=[p_step, z_step, z_step],
            axes_origin=[1.0, epsilon, epsilon],
            epsilon_z=epsilon,
            extrapolation_flag=True,
        )
        self.physics.add_property_region(pc)

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1))

    def _initial_profiles(self):
        nx = self.reservoir.nx
        x = (np.arange(nx) + 0.5) / nx          # normalized cell centres in (0,1)
        z_co2_bg, z_c1_bg = self.z_bg
        z_co2 = np.full(nx, z_co2_bg)
        if self.regime == "smooth":
            # compact smooth cosine^2 bump centred at x=0.3, width 0.25
            centre, half = 0.30, 0.125
            mask = np.abs(x - centre) < half
            z_co2[mask] += self.bump_amp * np.cos(np.pi * (x[mask] - centre) / (2 * half)) ** 2
        elif self.regime == "ramp":
            # smooth MONOTONE tanh transition (no interior extremum) centred at
            # x=0.30 -> cleanest probe of formal 2nd-order (WENO-JS keeps optimal
            # order away from critical points).
            width = 0.06
            z_co2 = z_co2 + self.bump_amp * 0.5 * (1.0 - np.tanh((x - 0.30) / width))
        elif self.regime == "sharp":
            # step: high CO2 upstream (x<0.3), background downstream
            z_co2[x < 0.30] += self.bump_amp
        else:
            raise ValueError(self.regime)
        z_c1 = np.full(nx, z_c1_bg)
        return z_co2, z_c1

    def set_initial_conditions(self):
        z_co2, z_c1 = self._initial_profiles()
        input_distribution = {
            self.physics.vars[0]: 50.0,      # pressure
            self.physics.vars[1]: z_co2,
            self.physics.vars[2]: z_c1,
        }
        return self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=input_distribution
        )

    def set_well_controls(self):
        # Inject the BACKGROUND composition at the inlet so there is NO
        # compositional front introduced at the well; the interior bump/step is
        # what advects.  Drive a uniform flow with a fixed pressure drop.
        z_co2_bg, z_c1_bg = self.z_bg
        inj_composition = [z_co2_bg, z_c1_bg]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(
                    wctrl=w.control, control_type=well_control_iface.BHP,
                    is_inj=True, target=60.0, inj_composition=inj_composition,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=w.control, control_type=well_control_iface.BHP,
                    is_inj=False, target=50.0,
                )
