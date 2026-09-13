"""1D elastic wave propagation in a column (thesis Sec. 6.4.1).

Column [0,1] x [0,1] x [0,10] m, E = 1 GPa, nu = 0.25, total density
rho = 2406 kg/m3, rollers on all boundaries, prescribed normal displacement
u_z(t) at the top: a rectangular compression pulse u_z0 = -0.01 m lasting
T = 0.001 s. Pure elasticity: flow and Biot coupling are switched off
(apply_geomechanics_mode(mode=2), biot = 0).
"""

import numpy as np

from darts.input.input_data import InputData
from darts.models.thmc_model import THMCModel
from darts.reservoirs.unstruct_reservoir_mech import (
    bound_cond,
    get_bulk_modulus,
    get_isotropic_stiffness,
    get_rock_compressibility,
)

from reservoir import ColumnReservoir

DAY = 86400.0  # seconds


class Model(THMCModel):
    def __init__(self, mesh_filename: str, E_pa: float = 1.e9, nu: float = 0.25, rho: float = 2406.0,
                 u_pulse: float = -0.01, t_pulse: float = 1.e-3, side_bc: str = 'roller'):
        self.physics_type = 'poromechanics'
        self.discretizer_name = 'pm_discretizer'
        self.mesh_filename = mesh_filename
        self.E_pa = E_pa
        self.nu = nu
        self.rho = rho
        self.u_pulse = u_pulse  # [m]
        self.t_pulse = t_pulse  # [s]
        # lateral faces: 'roller' (uniaxial strain, as in the thesis) or 'stuck' (clamped sides, for experiments only).
        # NOTE: on a one-cell-wide column the pm_discretizer roller stencils carry ~1e30 entries in the u_x/u_y
        # rows (degenerate lateral gradient reconstruction); use >= 2x2 cells across (main.py default).
        self.side_bc = side_bc
        super().__init__()

    # ------------------------------------------------------------------ physics
    @property
    def c_p(self) -> float:
        """P-wave (uniaxial-strain) velocity sqrt((lambda + 2 mu) / rho) [m/s]."""
        lam = self.E_pa * self.nu / (1. + self.nu) / (1. - 2. * self.nu)
        mu = self.E_pa / 2. / (1. + self.nu)
        return np.sqrt((lam + 2. * mu) / self.rho)

    def u_top(self, t_sec: float) -> float:
        """Prescribed top displacement: rectangular pulse."""
        return self.u_pulse if (t_sec > 0.0 and t_sec <= self.t_pulse * (1. + 1.e-9)) else 0.0

    def analytical_uz(self, z: np.ndarray, t_sec: float) -> np.ndarray:
        """d'Alembert solution for the downward travelling wave (before reflection at z = 0)."""
        H = self.reservoir.H
        tau = t_sec - (H - z) / self.c_p
        return np.array([self.u_top(ti) for ti in tau])

    # ------------------------------------------------------------------ setup
    def set_input_data(self):
        self.idata = InputData(type_hydr='isothermal', type_mech='poroelasticity', init_type='uniform')
        self.idata.other.case_name = 'elastic_wave_1d'

        self.idata.rock.density = self.rho  # kg/m3 (bookkeeping only; the engine uses momentum_inertia)
        self.idata.fluid.Mw = 18.015
        self.idata.fluid.density = self.idata.fluid.Mw
        self.idata.fluid.compressibility = 1.e-5
        self.idata.fluid.viscosity = 1.0
        self.idata.fluid.heat_capacity = 167.2  # not used (isothermal), required by InputData.check
        self.idata.fluid.thermal_conductivity = 0.

        self.bc_type = bound_cond()
        NO_FLOW = self.bc_type.NO_FLOW
        self.idata.mesh.bnd_tags = {'BND_X-': 991, 'BND_X+': 992, 'BND_Y-': 993, 'BND_Y+': 994,
                                    'BND_Z-': 995, 'BND_Z+': 996}
        bnd_tags = self.idata.mesh.bnd_tags
        self.idata.mesh.mesh_filename = self.mesh_filename
        self.idata.mesh.matrix_tags = [99991]

        self.idata.initial.initial_temperature = 0
        self.idata.initial.initial_pressure = 0
        self.idata.initial.initial_displacements = [0., 0., 0.]
        self.idata.initial.initial_composition = None

        self.idata.rock.porosity = 0.1
        self.idata.rock.perm = 1.0
        self.idata.rock.E = self.E_pa / 1.e5  # bars
        self.idata.rock.nu = self.nu
        self.idata.rock.biot = 0.0  # pure elasticity
        self.idata.rock.compressibility = get_rock_compressibility(
            kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu), biot=1.e-6, poro0=self.idata.rock.porosity)
        self.idata.rock.stiffness = get_isotropic_stiffness(self.idata.rock.E, self.idata.rock.nu)

        roller = {'flow': NO_FLOW, 'mech': self.bc_type.ROLLER}
        side = roller if self.side_bc == 'roller' else {'flow': NO_FLOW, 'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0])}
        self.idata.boundary = {}
        for key in ['BND_X-', 'BND_X+', 'BND_Y-', 'BND_Y+']:
            self.idata.boundary[bnd_tags[key]] = dict(side)
        self.idata.boundary[bnd_tags['BND_Z-']] = dict(roller)
        self.idata.boundary[bnd_tags['BND_Z+']] = {'flow': NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.0)}

        from darts.timestep_control import TimestepControl
        self.idata.sim.TimestepControl = TimestepControl(n_vars=0)
        self.idata.sim.time_steps = np.array([1.0])

        self.idata.obl.zero = 1e-9
        self.idata.obl.epsilon_z = 1e-10
        self.idata.obl.p_step = 1.0
        self.idata.obl.p_origin = -5.
        self.idata.obl.z_step = 2e-3
        self.idata.obl.z_origin = 0.
        self.idata.obl.t_step = 0.25
        self.idata.obl.t_origin = -10.
        super().set_input_data()

    def set_reservoir(self):
        self.reservoir = ColumnReservoir(timer=self.timer, idata=self.idata, fluid_vars=self.physics.vars)

    def set_solver(self):
        self.ts_control = self.idata.sim.TimestepControl
        super().set_solver()
        self.nonlinear_solver.spec.tolerance = 1e-8
        self.nonlinear_solver.spec.max_iterations = 20
