from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params
from darts.engines import well_control_iface, ms_well
from darts.nonlinear_solvers import NewtonSolver, ChopSpec

from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from dartsflash.mixtures import DARTSFlash, CompData, EoS, IAPWS
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.viscosity import MaoDuan2009


class Model(CICDModel):
    def __init__(self, resolution=10):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.resolution = resolution
        self.set_reservoir(resolution)
        self.set_physics()

        # solver configuration moved to set_solver() (called from base reset())

        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.set_sim_params(first_ts=1e-6, mult_ts=8, max_ts=31, runtime=365  )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-4, max_iterations=20,
            chop=ChopSpec(mode='global', factor=0.2))
        self.linear_solver.spec.tolerance = 1e-6
        self.linear_solver.spec.max_iterations = 40

    def set_reservoir(self, resolution):
        y_scale = 3
        (nx, ny, nz) = (resolution, y_scale * resolution, resolution)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000

        poro = np.ones(nb) * 0.2
        dx = 20. / resolution
        dy = 20. / resolution
        dz = 20. / resolution

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000, hcap=2200, rcond=500
                                         )
        self.reservoir.boundary_volumes['xy_minus'] = 1e8
        self.reservoir.boundary_volumes['xy_plus'] = 1e8
        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        return

    def set_wells(self):
        # add well's start locations
        iw = [self.resolution // 2, self.resolution // 2]
        jw = [4, self.reservoir.ny - 4]

        n = self.reservoir.nz // 2
        j_mid = self.reservoir.ny // 2

        well_diameter = 0.6

        # add well
        self.reservoir.add_well("INJ")
        for j in range(jw[0], j_mid + 1):
            self.reservoir.add_perforation("INJ", res_cell_idx=(iw[0], j, n + 1), well_diameter=well_diameter,
                                           segment_direction='y_axis', well_index=0, ms_epm=True)
        perf_1 = len(self.reservoir.wells[-1].perforations)  # last segment is n_perf+1

        self.reservoir.add_well("PRD")
        for j in range(jw[1], j_mid, -1):
            self.reservoir.add_perforation("PRD", res_cell_idx=(iw[1], j, n + 1), well_diameter=well_diameter,
                                           segment_direction='y_axis', well_index=0, ms_epm=True)
        perf_2 = len(self.reservoir.wells[-1].perforations)

        # connect the last two perforations of two wells
        # dictionary: key is a pair of 2 well names; value is a list of well perforation indices to connect
        self.reservoir.connected_well_segments = {
            (self.reservoir.wells[0].name, self.reservoir.wells[1].name): [(perf_1, perf_2)]
        }

    def set_physics(self):
        # create compositional + IAPWS PT-flash physics (drop-in for legacy Geothermal).
        # OBL cell sizes reproduce the former 128-point grid over p in [1, 351] bar and
        # T in [273.15, 575] K; the adaptive interpolator extends past it on demand.
        self.set_iapws_physics(p_step=2.756, p_origin=1.,
                               t_step=2.377, t_origin=273.15, cache=False)

        return

    def set_iapws_physics(self, p_step, p_origin, t_step, t_origin, cache=False):
        """Drop-in replacement for legacy Geothermal(...) using compositional + IAPWS PT-flash."""
        components = ["H2O"]
        phases     = ['V', 'L']           # vapor, liquid (replaces 'steam','water')
        zero       = 1e-12
        comp_data  = CompData(components=components, setprops=True)

        pc = PropertyContainer(phases_name=phases, components_name=components,
                               Mw=comp_data.Mw, eps_z=zero)

        flash_ev = IAPWS(iapws_ideal=True, ice_phase=False)
        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash)
        pc.flash_ev = flash_ev

        pc.density_ev = {
            'V': EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX),
            'L': EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MIN),
        }
        pc.viscosity_ev = {
            'V': ConstFunc(0.01),         # cP, steam
            'L': MaoDuan2009(components),  # cP, liquid water (pressure/temperature-dependent)
        }
        pc.enthalpy_ev = {
            'V': EoSEnthalpy(eos=flash_ev.eos["IAPWS"], root_flag=EoS.RootFlag.MAX),
            'L': EoSEnthalpy(eos=flash_ev.eos["IAPWS"], root_flag=EoS.RootFlag.MIN),
        }
        pc.rel_perm_ev = {
            'V': PhaseRelPerm("gas", swc=0.0),
            'L': PhaseRelPerm("oil", swc=0.0),
        }
        pc.conductivity_ev = {
            'V': ConstFunc(0.0),
            'L': ConstFunc(172.8),        # kJ/m/day/K, matches geothermal default
        }
        # output_props exposes derived T (K) via the property interpolator
        pc.output_props = {'temperature': lambda: pc.temperature}

        # Single component (H2O) with state_spec=PT -> OBL axes are [pressure, temperature]
        self.physics = PhysicsBase(
            components, phases, self.timer,
            state_spec=PhysicsBase.StateSpecification.PT,
            axes_step=[p_step, t_step],
            axes_origin=[p_origin, t_origin],
            epsilon_z=zero,
            cache=cache,
        )
        self.physics.add_property_region(pc)
        return pc

    def set_initial_conditions(self):
        input_distribution = {'pressure': 200.,
                              'temperature': 450.
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=205, phase_name='L',
                                               inj_composition=[], inj_temp=300.)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=195., phase_name='L')
