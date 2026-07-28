import numpy as np
import matplotlib.pyplot as plt
import os

from dataclasses import dataclass
from darts.models.darts_model import DartsModel
from darts.nonlinear_solvers import Norm, NewtonSolver, ChopSpec
from darts.engines import value_vector
from darts.input.input_data import linear_solver_types
from math import fabs
try:
    from darts.engines import copy_data_to_device, copy_data_to_host, allocate_device_data
except ImportError:
    pass
from darts.engines import well_control_iface
from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, CapillaryPressure, PhaseRelPerm
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData
from scipy.special import erf

# region Dataclasses
@dataclass
class Corey:
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    pcmax: float
    c2: float
    def modify(self, std, mult):
        i = 0
        for attr, value in self.__dict__.items():
            if attr != 'type':
                setattr(self, attr, value * (1 + mult[i] * float(getattr(std, attr))))
            i += 1

    def random(self, std):
        for attr, value in self.__dict__.items():
            if attr != 'type':
                std_in = value * float(getattr(std, attr))
                param = np.random.normal(value, std_in)
                if param < 0:
                    param = 0
                setattr(self, attr, param)


@dataclass
class PorPerm:
    type: str
    poro: float
    perm: float
    anisotropy: list = None
    hcap: float = 2125
    rcond: float = 181.44

# endregion

def build_output_dir(spec, base_dir=""):
    iso_tag = "iso" if spec.get("temperature") is not None else "niso"
    rhs_tag = "rhs" if spec.get("RHS") else "wells"
    disp_tag = "disp" if spec.get("dispersion") else "nodisp"
    components = "-".join(spec.get("components", []))
    nx = spec.get("nx", "nx?")
    ny = spec.get("ny", "ny?")
    nz = spec.get("nz", "nz?")
    device = spec.get("platform", "platform?")

    # Construct folder name
    dir_name = f"{iso_tag}__{rhs_tag}__{disp_tag}__{components}__nx{nx}_ny{ny}_nz{nz}_{device}"
    return os.path.join(base_dir, dir_name)

########
import pickle
from darts.engines import redirect_darts_output, sim_params

# For each of the facies within the SPE11b model we define a set of operators in the physics.
property_regions  = [0, 1, 2, 3, 4, 5, 6]
layers_to_regions = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}
######

class Model(DartsModel):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.components = self.specs['components']
        self.nc = len(self.components)
        self.zero = 1e-10
        self.salinity = 0

        """Define physics"""
        self.set_physics(temperature=specs['temperature'])
        # OBL is unbounded in this branch (axes_origin/axes_step, no clamp). In the
        # post-injection migration phase the Newton solver overshoots compositions toward
        # the simplex boundary (z->0/1); the unbounded interpolator then extrapolates into
        # unphysical state (z<0, with cascading p/T excursions that reach t<0 K -> NaN),
        # which stalls Newton ("stationary point") and triggers many timestep cuts (~2x
        # slower migration phase vs the bounded reference). A *local* (composition) chop
        # caps |dz| per Newton step and removes those stalls: newton_local_chop[0.01]
        # reproduces the bounded-baseline timestep/cut counts and runtime. (The previous
        # global chop uses relative |dX|/|X|, which over-restricts near z~1e-11 and did
        # not prevent the cuts; looser local caps >=0.1 let the solver reach t<0 K -> NaN.)
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=12,
                                           chop=ChopSpec(mode='local', factor=0.01),
                                           norm=Norm.L2)  # Norm.LINF if you use m.set_rhs() for injection
        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=365, tol_linear=1e-4,
                            it_linear=50)
        # self.data_ts.eta = np.ones(self.physics.n_vars)

        """ Define reservoir """
        self.set_reservoir()

        """ Define initial and boundary conditions """
        self.inj_stream = specs['inj_stream'] # define injection stream of the wells
        inj_rate = specs['inj_rate']  # mass rate per well, kg/day
        if specs['1000years'] is False:
            self.inj_rate = [inj_rate, self.zero]
        else:
            self.inj_rate = [0, 0]

        if specs['platform'] == 'cpu':
            self.platform = 'cpu'
            try:
                from darts.engines import set_num_threads
                set_num_threads(16)
            except:
                pass

        elif specs['platform'] == 'gpu':
            self.platform = 'gpu'
            from darts.engines import set_gpu_device
            set_gpu_device(0)

    def set_reservoir(self):
        """ Define the reservoir and wells """

        if self.specs['ny'] == 1:

            cmult = 86.4
            layer_props = {900001: PorPerm(type='7', poro=1e-6, perm=1e-6, anisotropy=[1, 1, 0.1], rcond=2.0 * cmult),
                           900002: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
                           900003: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
                           900004: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
                           900005: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
                           900006: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
                           900007: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
                           900008: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
                           900009: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
                           900010: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900011: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900012: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900013: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900014: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900015: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900016: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900017: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900018: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900019: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900020: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900021: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900022: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900023: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1], rcond=0.26 * cmult),
                           900024: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1], rcond=0.26 * cmult),
                           900025: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1], rcond=0.26 * cmult),
                           900026: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900027: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900028: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900029: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900030: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           900031: PorPerm(type='7', poro=1e-6, perm=1e-6, anisotropy=[1, 1, 0.1], rcond=2.0 * cmult),
                           900032: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
                           }

            well_centers = {
                "I1": [2700.0, 0.0, 300.0],
                "I2": [5100.0, 0.0, 700.0]
            }

            if 1:
                # ---- SPE11b
                from fluidflower_str_b import FluidFlowerStruct
                self.reservoir = FluidFlowerStruct(timer=self.timer, layer_properties=layer_props,
                                                    layers_to_regions=layers_to_regions,
                                                        model_specs=self.specs, well_centers=well_centers)

            else:
                # ---- Homogeneous version of SPE11b
                from fluidflower_str_b_homogeneous import FluidFlowerStruct
                self.reservoir = FluidFlowerStruct(timer=self.timer, layer_properties=layer_props,
                                                        layers_to_regions=layers_to_regions,
                                                            model_specs=self.specs, well_centers=well_centers)

            self.nx, self.ny, self.nz = self.reservoir.nx, self.reservoir.ny, self.reservoir.nz
            self.grid = np.meshgrid(np.linspace((8400 / self.nx / 2), 8400 - (8400 / self.nx / 2), self.nx),
                               np.linspace((1200 / self.nz / 2), 1200 - (1200 / self.nz / 2), self.nz))

            self.set_str_boundary_volume_multiplier()  # right and left boundary volume multiplier

        elif self.specs['ny'] != 1:
            # ---- SPE11c
            dmult, cmult = 0.9869233, 86.4
            layer_props = {1: PorPerm(type='1', poro=0.1, perm=0.1 / dmult, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
                           2: PorPerm(type='2', poro=0.2, perm=100 / dmult, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           3: PorPerm(type='3', poro=0.2, perm=200 / dmult, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           4: PorPerm(type='4', poro=0.2, perm=500 / dmult, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
                           5: PorPerm(type='5', poro=0.25, perm=1000 / dmult, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
                           6: PorPerm(type='6', poro=0.35, perm=2000 / dmult, anisotropy=[1, 1, 0.1], rcond=0.26 * cmult),
                           7: PorPerm(type='7', poro=1.e-5, perm=1.e-6, anisotropy=[1, 1, 0.1], rcond=2.0 * cmult)
                           }

            well_geometry = {
                'I1': {'head': np.array([2700, 1000, 300]), 'tail': np.array([2700, 4000, 300]), 'coordinates': 'cartesian'}, # well 1
                'I2': {'head': np.array([5100, 1000, 700]), 'tail': np.array([5100, 4000, 700]), 'coordinates': 'reference'}, # well 2
            }

            mesh_file_name = (
                f"spe11c_structured_"
                f"{self.specs['nx']}_"
                f"{self.specs['ny']}_"
                f"{self.specs['nz']}.msh"
            )

            if os.path.exists(mesh_file_name):
                mesh_file = mesh_file_name
            else:
                import subprocess

                subprocess.run(
                    [
                        "python",
                        "make_structured_mesh.py",
                        "-v", "C",
                        "-nx", str(self.specs['nx']),
                        "-ny", str(self.specs['ny']),
                        "-nz", str(self.specs['nz']),
                    ],
                    check=True
                )

                default_name = "spe11c_structured.msh"

                if os.path.exists(default_name):
                    os.rename(default_name, mesh_file_name)
                    mesh_file = mesh_file_name
                else:
                    raise FileNotFoundError("Mesh generation failed.")

            from fluidflower_new_str_c import FluidFlowerNewStruct
            self.reservoir = FluidFlowerNewStruct(self.specs, timer=self.timer, layer_props=layer_props,
                                                  well_geometry=well_geometry, mesh_file=mesh_file)

            # from darts.reservoirs.unstruct_reservoir import UnstructReservoir
            # permx = 100.
            # permy = 100.
            # permz = 100.
            # poro = 0.2
            # self.reservoir = UnstructReservoir(self.timer, mesh_file, permx, permy, permz, poro, op_num = 1)
            # self.reservoir.physical_tags['matrix'] = [1, 2, 3, 4, 5, 6, 7]
            # # self.set_boundary_conditions_11c()


    def set_wells(self):
        self.reservoir.set_wells(False)
        return

    def set_well_controls(self):
        if self.specs['RHS'] is False:
            for i, w in enumerate(self.reservoir.wells):
                self.physics.set_well_controls(wctrl = w.control,
                                               control_type = well_control_iface.MASS_RATE,
                                               is_inj = True,
                                               target = self.inj_rate[i],
                                               phase_name = 'V',
                                               inj_composition = self.inj_stream[:-1],
                                               inj_temp = self.inj_stream[-1]
                                               )
                print(f'Set well {w.name} to {self.inj_rate[i]} kg/day with {self.inj_stream[:-1]} {self.components[:-1]} at 10°C...')
            return
        else:
            pass

    def apply_rhs_flux(self, dt: float, t: float):
        if self.specs['RHS'] is False:
            # If the function has not been overloaded, pass
            return
        rhs = np.array(self.physics.engine.RHS, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs[:n_res] += self.set_rhs_flux(t) * dt
        return

    def set_rhs_flux(self, t: float = None):
        if self.specs['RHS'] is True:
            nc = self.physics.nc
            nv = self.physics.n_vars
            nb = self.reservoir.mesh.n_res_blocks
            rhs = np.zeros(nb * nv)

            region = 0
            molar_masses = self.physics.property_containers[region].Mw
            mole_fractions = self.inj_stream[:nc]
            n_comp = np.zeros(nc)
            nu_idxV = list(self.physics.property_containers[region].output_props.keys()).index("nu_V")
            nu_idxA = list(self.physics.property_containers[region].output_props.keys()).index("nu_Aq")
            enth_idx = list(self.physics.property_containers[region].output_props.keys()).index("enthalpy_V")
            enth_idxAq = list(self.physics.property_containers[region].output_props.keys()).index("enthalpy_Aq")

            for i, well in enumerate(self.reservoir.well_cells):
                for well_cell in well:
                    p_wellcell = self.physics.engine.X[well_cell * nv]
                    if self.physics.thermal:
                        state = value_vector([p_wellcell, *self.inj_stream[:-2], self.inj_stream[-1]])
                    else:
                        state = value_vector([p_wellcell] + self.inj_stream[:-2])

                    values = value_vector(np.zeros(self.physics.n_ops))
                    # values_np = np.array(values)
                    self.physics.property_itor[self.op_num[well_cell]].evaluate(state, values)
                    enth = values[nu_idxV] * values[enth_idx] + values[nu_idxA] * values[enth_idxAq]  # mole fraction moles in vapour [V/V+A] * molar enthalpy of vapour [kJ/kmol] + aq
                    # enth = self.physics.property_containers[0].compute_total_enthalpy(state)
                    avg_molar_mass = sum(mf * M for mf, M in zip(mole_fractions, molar_masses))
                    tot_moles = self.inj_rate[i] / avg_molar_mass / len(well)  # kg/day / kg/mol -> mol/day

                    for comp_idx in range(nc):
                        comp_flux_idx = well_cell * nv + comp_idx  # Index
                        n_comp[comp_idx] = tot_moles * mole_fractions[comp_idx]  # Compute component moles
                        rhs[comp_flux_idx] -= n_comp[comp_idx]  # Update rhs

                    if self.physics.thermal:
                        temp_idx = well_cell * nv + nv - 1  # Last equation index (temperature)
                        rhs[temp_idx] -= enth * tot_moles

            return rhs

    def set_physics(self, temperature: float = None):
        """Physical properties"""

        # define the Corey parameters for each layer (rock type) according to the technical description of the CSP
        corey = {
            0: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1.935314, pcmax=300, c2=1.5),
            1: Corey(nw=1.5, ng=1.5, swc=0.14, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.08655, pcmax=300, c2=1.5),
            2: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0612, pcmax=300, c2=1.5),
            3: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.038706, pcmax=300, c2=1.5),
            4: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0306, pcmax=300, c2=1.5),
            5: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300, c2=1.5),
            6: Corey(nw=1.5, ng=1.5, swc=1e-8, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1e-2, pcmax=300, c2=1.5)
        }

        """Physical properties"""

        from dartsflash.libflash import EoS
        from dartsflash.components import CompData
        from dartsflash.mixtures import DARTSFlash, VLAq
        # Fluid components, ions and solid
        phases = ["V", "Aq"]
        comp_data = CompData(self.components, setprops=True)
        nc = len(self.components)

        """ Define flash """
        flash_ev = VLAq(comp_data, hybrid=True)
        flash_ev.set_vl_eos("PR", root_order=[EoS.STABLE],
                            trial_comps=[i for i in range(nc)],
                            stability_tol=1e-20, switch_tol=1e-2, max_iter=50, use_gmix=False
                            )
        flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
        pr = flash_ev.eos["VL"]
        aq = flash_ev.eos["Aq"]

        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["VL", "Aq"],
                            t_min=270., t_max=500., t_init=300.,
                            # pxflash_switch_ttol=1e-3, near_zero_px=1e-2,
                            )

        if temperature is None:  # if None, then thermal=True
            thermal = True
            state_spec = PhysicsBase.StateSpecification.PT
        else:
            thermal = False
            state_spec = PhysicsBase.StateSpecification.P

        pres_in = 210 # (pressure at depth of well 1 will be 300 bar)
        # 1 p axis + (nc-1) z axes + optional T axis
        nz = len(self.components) - 1
        ax_step = [0.25] + [1e-3] * nz
        ax_origin = [200.0] + [self.zero / 10] * nz
        if thermal:
            ax_step.append(0.1)
            ax_origin.append(273.15)
        self.physics = PhysicsBase(self.components, phases, timer=self.timer,
                                     axes_step=ax_step, axes_origin=ax_origin,
                                     epsilon_z=self.zero / 10,
                                     state_spec=state_spec,
                                     extrapolation_flag=False,
                                     cache=False)

        dispersivity = 10.
        self.physics.dispersivity = {}

        for i, (region, corey_params) in enumerate(corey.items()):
            diff_w = 1e-9 * 86400
            diff_g = 2e-8 * 86400
            property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                                   eps_z=self.zero / 10, temperature=temperature)

            property_container.flash_ev = flash_ev
            property_container.density_ev = dict([('V', EoSDensity(eos=flash_ev.eos["VL"], Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(self.components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(self.components)), ])
            property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff_g)),
                                                    ('Aq', ConstFunc(np.ones(nc) * diff_w))])
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=flash_ev.eos["VL"])),
                                                   ('Aq', EoSEnthalpy(eos=flash_ev.eos["Aq"]))])
            property_container.conductivity_ev = dict([('V', ConstFunc(8.4)),
                                                       ('Aq', ConstFunc(170.)),])
            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)

            self.physics.add_property_region(property_container, i)

            property_container.output_props = {
                "sat_V": lambda ii=i: self.physics.property_containers[ii].sat[phases.index('V')],
                "dens_V": lambda ii=i: self.physics.property_containers[ii].dens[phases.index('V')],
                "densm_Aq": lambda ii=i: self.physics.property_containers[ii].dens_m[phases.index('Aq')],
                "enthalpy_V": lambda ii=i: self.physics.property_containers[ii].enthalpy[phases.index('V')],
                "enthalpy_Aq": lambda ii=i: self.physics.property_containers[ii].enthalpy[phases.index('Aq')],
                "nu_V": lambda ii=i: self.physics.property_containers[ii].nu[phases.index('V')],
                "nu_Aq": lambda ii=i: self.physics.property_containers[ii].nu[phases.index('Aq')],
                }

            for j, phase_name in enumerate(phases):
                for c, component_name in enumerate(self.components):
                    key = f"x_{phase_name}_{component_name}"
                    property_container.output_props[key] = lambda ii=i, jj=j, cc=c: self.physics.property_containers[ii].x[jj, cc]

            if region == 0 or region == 6:
                self.physics.dispersivity[region] = np.zeros((self.physics.nph, self.physics.nc))
            else:
                disp = dispersivity * np.ones((self.physics.nph, self.physics.nc))
                disp[0, :] /= diff_g
                disp[1, :] /= diff_w
                self.physics.dispersivity[region] = disp

    def init_dispersion(self):
        # activate reconstruction of velocities
        self.reconstruct_velocities()

        # set dispersion coefficients
        nph = self.physics.nph
        nc = self.physics.nc
        self.physics.engine.dispersivity.resize(len(self.physics.regions) * self.physics.nph * self.physics.nc)
        dispersivity = np.asarray(self.physics.engine.dispersivity)
        for i, region in enumerate(self.physics.regions):
            dispersivity[i * nph * nc:(i + 1) * nph * nc] = self.physics.dispersivity[region].flatten()

        print('Fickian energy is', self.physics.engine.is_fickian_energy_transport_on)

        # allocate & transfer dispersivities to device
        if self.platform == 'gpu':
            dispersivity_d = self.physics.engine.get_dispersivity_d()
            allocate_device_data(self.physics.engine.dispersivity, dispersivity_d)
            copy_data_to_device(self.physics.engine.dispersivity, dispersivity_d)

    def reconstruct_velocities(self):
        # velocity discretization
        values, offset = self.reservoir.discretizer.discretize_velocities(
            cell_m=np.asarray(self.reservoir.mesh.block_m),
            cell_p=np.asarray(self.reservoir.mesh.block_p),
            geom_coef=np.asarray(self.reservoir.mesh.tranD),
            n_res_blocks=self.reservoir.mesh.n_res_blocks,
        )
        self.reservoir.mesh.velocity_appr.resize(len(values))
        self.reservoir.mesh.velocity_offset.resize(len(offset))

        velocity_appr = np.asarray(self.reservoir.mesh.velocity_appr)
        velocity_appr[:] = values #/ (8400. / self.reservoir.nx * 1.)
        velocity_offset = np.asarray(self.reservoir.mesh.velocity_offset)
        velocity_offset[:] = offset

        # specify molar weights to get rid of molar density multiplier in flux terms
        nc = self.physics.nc
        self.physics.engine.molar_weights.resize(nc * len(self.physics.regions))
        molar_weights = np.asarray(self.physics.engine.molar_weights)
        for i, region in enumerate(self.physics.regions):
            molar_weights[i * nc : (i + 1) * nc] = self.physics.property_containers[
                region
            ].Mw

        # resize storage for velocities inside engine
        self.physics.engine.darcy_velocities.resize(
            self.reservoir.mesh.n_res_blocks * self.physics.nph * 3
        )

        # allocate & transfer data to device
        if self.platform == "gpu":
            from darts.engines import allocate_device_data, copy_data_to_device

            # velocity_appr
            velocity_appr_d = self.physics.engine.get_velocity_appr_d()
            allocate_device_data(self.reservoir.mesh.velocity_appr, velocity_appr_d)
            copy_data_to_device(self.reservoir.mesh.velocity_appr, velocity_appr_d)
            # velocity_offset_d
            velocity_offset_d = self.physics.engine.get_velocity_offset_d()
            allocate_device_data(self.reservoir.mesh.velocity_offset, velocity_offset_d)
            copy_data_to_device(self.reservoir.mesh.velocity_offset, velocity_offset_d)
            # darcy_velocities_d
            darcy_velocities_d = self.physics.engine.get_darcy_velocities_d()
            allocate_device_data(
                self.physics.engine.darcy_velocities, darcy_velocities_d
            )
            # molar_weights_d
            molar_weights_d = self.physics.engine.get_molar_weights_d()
            allocate_device_data(self.physics.engine.molar_weights, molar_weights_d)
            copy_data_to_device(self.physics.engine.molar_weights, molar_weights_d)
            # op_num_d
            op_num_d = self.physics.engine.get_op_num_d()
            allocate_device_data(self.reservoir.mesh.op_num, op_num_d)
            copy_data_to_device(self.reservoir.mesh.op_num, op_num_d)

    def set_initial_conditions(self):

        if self.specs['ny'] != 1:

            temp = lambda depth : (273.15 + 70) - depth * 0.025
            pres = lambda depth : 212 + depth * 0.09775

            if 0:
                z_depths = [np.amin(self.reservoir.centroids[:, 2]), np.amax(self.reservoir.centroids[:, 2])]
                input_depths = [np.amin(self.reservoir.mesh.depth), np.amax(self.reservoir.mesh.depth)]

                input_distribution = {}
                input_distribution['CO2'] = [self.zero, self.zero]
                input_distribution['H2O'] = [1. - self.zero, 1 - self.zero]
                input_distribution['pressure'] = [pres(depth) for depth in z_depths]
                input_distribution['temperature'] = [temp(depth) for depth in z_depths[::-1]]

                self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                                     input_depth=input_depths,
                                                                     input_distribution=input_distribution)
            else:
                input_distribution = {}
                input_distribution['CO2'] = self.zero
                input_distribution['H2O'] = 1. - self.zero

                self.reservoir.centroids = self.reservoir.discretizer.centroid_all_cells
                z_depths = self.reservoir.centroids[:, 2]

                input_distribution['pressure'] = np.array([pres(depth) for depth in z_depths[::-1]])
                input_distribution['temperature'] = np.array([temp(depth) for depth in z_depths])

                self.physics.set_initial_conditions_from_array(self.reservoir.mesh, input_distribution)

        if self.specs['ny'] == 1:
            if 1:
                pres_in = 212
                input_depths = [np.amin(self.reservoir.mesh.depth), np.amax(self.reservoir.mesh.depth)]

                self.input_distribution = {"pressure": [pres_in, pres_in + input_depths[1] * 0.09775]}
                for i in range(self.nc):
                    if self.components[i] == 'H2O':
                        self.input_distribution[self.components[i]] = [1-(self.nc-1)*self.zero, 1-(self.nc-1)*self.zero]
                    else:
                        self.input_distribution[self.components[i]] = [self.zero, self.zero]

                if self.specs['temperature'] is None:
                    bot_cell = self.reservoir.bot_cells[0]
                    T_spec_bot = 273.15 + 70 - self.reservoir.centroids[bot_cell, 2] * 0.025

                    top_cell = self.reservoir.top_cells[0]
                    T_spec_top = 273.15 + 70 - self.reservoir.centroids[top_cell, 2] * 0.025

                    self.input_distribution["temperature"] = [T_spec_top, T_spec_bot]

                self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                                     input_depth=input_depths,
                                                                     input_distribution=self.input_distribution)

            if 0:
                self.temp = lambda depth: 273.15 + 70. - depth * 0.025

                depths = np.asarray(self.reservoir.mesh.depth)
                min_depth = np.min(depths)
                max_depth = np.max(depths)
                nb = 100
                depths = np.linspace(min_depth, max_depth, nb)

                # zH2O = 1
                from darts.physics.base.initialize import Initialize
                init = Initialize(self.physics, aq_idx=0, h2o_idx=0)
                nc = len(self.components)

                # Solve boundary state
                min_depth = self.reservoir.global_data['depth'].min()
                max_depth = self.reservoir.global_data['depth'].max()

                # Known conditions at well I1
                known_depth = self.reservoir.well_centers["I1"][2]
                pres_I1 = 300.
                temp_I1 = self.temp(max_depth - known_depth)  # depths in grid have z=0 at the bottom

                specs = {'pressure': pres_I1, 'temperature': temp_I1 if init.thermal else None}
                if nc == 2:
                    # H2O-CO2, initially pure brine
                    # need 1 specification: H2O = 1-zero
                    # specs["H2O"] = 1.-self.zero
                    specs["CO2"] = self.zero

                else:
                    # H2O-CO2-C1, initially pure brine
                    # need 2 specifications: H2O = 1-(nc-1)*zero, CO2 = zero
                    specs["H2S"] = self.zero
                    specs["CO2"] = self.zero

                if self.salinity:
                    # + ions, need extra specification for ion molality
                    # H2O cannot be specified because of salinity, last component instead
                    mol = self.salinity
                    specs.update({'m' + str(nc): mol})
                    specs.update({'m' + str(nc): mol})
                    specs["H2O"] = None
                    specs[self.components[-1]] = self.zero

                X0 = ([pres_I1, 0.98] +  # pressure, H2O
                      ([self.zero] if nc > 2 else []) +  # CO2
                      ([1. - 0.98 - mol * 0.98 / 55.509] if self.salinity else []) +  # last component if ions
                      ([temp_I1] if init.thermal else []))  # temperature
                X0 = init.solve_state(Xi=X0,
                                      specs=specs,
                                      )

                # Initialize depth table
                nb = 100
                X, bc_idx = init.init_depth_table(depth_bottom=max_depth,
                                                  depth_top=min_depth,
                                                  depth_known=known_depth,
                                                  X0=X0,
                                                  nb=nb,
                                                  dTdh=0.025
                                                  )

                # Solve vertical equilibrium
                specs['pressure'] = None  # set to None because pressure will be calculated from hydrostatic column
                X = init.solve(X=X, bc_idx=bc_idx, specs=specs, downward=False)  # solve above
                X = init.solve(X=X, bc_idx=bc_idx, specs=specs, downward=True)  # solve below

                self.physics.set_initial_conditions_from_depth_table(mesh = self.reservoir.mesh,
                                                                     input_depth = init.depths,
                                                                     input_distribution = {v: X[:, i] for i, v in enumerate(self.physics.vars)})

    def set_boundary_conditions(self):
        if self.specs['ny'] != 1:
            volume = np.array(self.reservoir.mesh.volume, copy=False)
            points = self.reservoir.mesh_data.points
            min_coord, max_coord = np.min(points, axis=0)[:2], np.max(points, axis=0)[:2]
            cell_points = points[self.reservoir.mesh_data.cells[0].data]
            x_minus_ids = np.unique(np.where(cell_points[:, :, 0] == min_coord[0])[0])
            x_plus_ids = np.unique(np.where(cell_points[:, :, 0] == max_coord[0])[0])
            y_minus_ids = np.unique(np.where(cell_points[:, :, 1] == min_coord[1])[0])
            y_plus_ids = np.unique(np.where(cell_points[:, :, 1] == max_coord[1])[0])
            self.side_cell_ids = np.unique(np.concatenate((x_minus_ids, x_plus_ids, y_minus_ids, y_plus_ids)))
            volume[self.side_cell_ids] *= 1e+6 # 5e4 * (1200 / self.reservoir.nz) * (5000 / self.reservoir.ny)

            # top and bottom
            if self.physics.thermal:
                pillar_ids = np.unique(self.reservoir.discretizer.centroid_all_cells[:, :2], axis=0, return_inverse=True)[1]
                self.reservoir.top_cells = []
                self.reservoir.bot_cells = []
                for i in range(np.max(pillar_ids) + 1):
                    ids = np.where(pillar_ids == i)[0]
                    self.reservoir.top_cells.append(ids[self.reservoir.discretizer.centroid_all_cells[ids, 2].argmax()])
                    self.reservoir.bot_cells.append(ids[self.reservoir.discretizer.centroid_all_cells[ids, 2].argmin()])
        else:
            pass
        return

    def set_str_boundary_volume_multiplier(self):
        self.reservoir.boundary_volumes['yz_minus'] = 5e9 * (1200 / self.reservoir.nz)
        self.reservoir.boundary_volumes['yz_plus']  = 5e9 * (1200 / self.reservoir.nz)
        return

    def get_mass_components(self, property_array):
        nb = self.reservoir.mesh.n_res_blocks
        component_names = self.physics.property_containers[0].components_name
        Mw = np.array(self.physics.property_containers[0].Mw).reshape(-1, 1)

        # Extract properties from property_array
        sg = property_array['sat_V'][0]
        rhoV = property_array['dens_V'][0]
        rho_m_Aq = property_array['densm_Aq'][0]

        self.x_components, self.y_components = [], []
        # for phase_name in self.physics.phases:
        for component_name in self.physics.components:
            self.x_components.append(property_array[f'x_Aq_{component_name}'][0])
            self.y_components.append(property_array[f'x_V_{component_name}'][0])
            # self.x_components.append(property_array['x' + component_name][0])
            # self.y_components.append(property_array['y' + component_name][0])

        self.x_components, self.y_components = np.array(self.x_components), np.array(self.y_components)

        # Compute molecular weight of the aqueous phase
        MWAq = np.sum(self.y_components[1:, :] * Mw[1:], axis = 0 ) + (1 - np.sum(self.y_components[1:, :], axis = 0)) * Mw[0]

        # Mass fractions in vapor phase
        w_components_vapor = (self.y_components * Mw) / MWAq

        # Pore volume

        V = np.array(self.reservoir.mesh.volume, copy=False)[:nb]
        if self.specs['ny'] != 1:
            V[self.side_cell_ids] /= 1e+6
        phi = np.array(self.reservoir.mesh.poro, copy=False)[:nb]

        # Calculate total mass for each component
        mass_components = {}
        mass_aqueous = {}
        mass_vapor = {}
        for i, component_name in enumerate(component_names):
            # Vapor phase mass contribution
            mass_vapor[component_name] = phi * V * w_components_vapor[i] * sg * rhoV

            # Aqueous phase mass contribution
            mass_aqueous[component_name] = phi * V * (1 - sg) * self.x_components[i] * rho_m_Aq * Mw[i]

            # Total mass
            mass_components[component_name] = mass_vapor[component_name] + mass_aqueous[component_name]

        return mass_components, mass_vapor, mass_aqueous

    def set_top_bot_temp(self):
        nv = self.physics.n_vars
        for bot_cell in self.reservoir.bot_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_bot = 273.15 + 70 - self.reservoir.centroids[bot_cell, 2] * 0.025
            target_cell = bot_cell*nv+nv-1
            self.physics.engine.X[target_cell] = T_spec_bot

        for top_cell in self.reservoir.top_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_top = 273.15 + 70 - self.reservoir.centroids[top_cell, 2] * 0.025
            target_cell = top_cell*nv+nv-1
            self.physics.engine.X[target_cell] = T_spec_top
        return


    def run_timestep(self, dt: float, t: float, verbose: bool = True):
        """
        Method to solve Newton loop for specified timestep

        :param dt: Timestep size [days]
        :type dt: float
        :param t: Current time [days]
        :type t: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        assert dt > 0, "Time step size must be a positive value!"

        max_newt = self.nonlinear_solver.spec.max_iterations
        max_residual = np.zeros(max_newt + 1)
        solver = self.nonlinear_solver
        status = solver.status
        status.reset()
        self.nonlinear_solver.spec.well_tolerance_multiplier = 1e2
        self.timer.node["simulation"].start()

        residual_history = []
        for i in range(max_newt + 1):

            if self.physics.thermal and self.specs['RHS'] is True:
                # apply bottom and top boundary conditions

                if self.platform == 'gpu':
                    copy_data_to_host(self.physics.engine.X, self.physics.engine.get_X_d())

                self.set_top_bot_temp()

                if self.platform == 'gpu':
                    copy_data_to_device(self.physics.engine.X, self.physics.engine.get_X_d())

            # Update well phase velocities and derivatives if DFM wells are used
            if self.has_dfm_well:
                self.update_dfm_well_vels_and_ders(dt, t, i)

            # assemble Jacobian and residual of reservoir and well blocks
            self.physics.engine.assemble_linear_system(dt)

            # apply RHS flux
            self.apply_rhs_flux(dt, t)


            if self.platform == "gpu":
                copy_data_to_device(
                    self.physics.engine.RHS, self.physics.engine.get_RHS_d()
                )

            if not self.has_dfm_well:
                status.newton_residual = (
                    self.physics.engine.calc_newton_residual()
                )  # calc norm of residual
            else:
                # Method is either 1 or 2
                status.newton_residual = (
                    self.physics.engine.calc_coupled_well_reservoir_residual(
                        self.nonlinear_solver.spec.coupled_well_res_norm_method
                    )
                )

            max_residual[i] = status.newton_residual
            counter = 0
            for j in range(i):
                denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                if (
                    abs(max_residual[i] - max_residual[j]) / denom
                    < self.nonlinear_solver.spec.stationary_point_tolerance
                ):
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            status.well_residual = self.physics.engine.calc_well_residual()
            residual_history.append(
                (
                    status.newton_residual,  # matrix residual
                    status.well_residual,  # well residual
                    1.0,
                )
            )  # Newton update coefficient

            status.n_newton = i
            #  check tolerance if it converges
            if (
                status.newton_residual < self.nonlinear_solver.spec.tolerance
                and status.well_residual
                < self.nonlinear_solver.spec.tolerance * self.nonlinear_solver.spec.well_tolerance_multiplier
            ) or status.n_newton == max_newt:
                if i > 0:  # min_i_newton
                    break

            if isinstance(self.data_ts.linear_type, linear_solver_types):
                # solvers via Python interface
                if self.data_ts.linear_type in [
                    linear_solver_types.CPU_PETSC_CPR,
                    linear_solver_types.CPU_PETSC_FS,
                ]:
                    self.petsc_solve_linear_equation()
                elif self.data_ts.linear_type in [linear_solver_types.CPU_PARDISO]:
                    self.pardiso_solve_linear_equation()
                else:
                    raise Exception(
                        "Unknown linear solver type", self.data_ts.linear_type
                    )
            else:
                # compile-time C++ linear solvers
                r_code = self.physics.engine.solve_linear_equation()
                status.linear_solver_rc = r_code
                if r_code != 0:
                    # failed linear solve: do NOT apply a stale update; the
                    # post-loop verdict reads status.linear_solver_rc -> fail
                    self._linear_solver_rc_last = r_code
                    break
                status.n_linear += self.physics.engine.get_last_linear_iters()
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()

        # End of newton loop: convergence verdict previously made by the C++
        # post_newtonloop (linear solver rc + residual re-check), now in Python.
        converged = not (
            status.linear_solver_rc != 0
            or status.newton_residual >= self.nonlinear_solver.spec.tolerance
            or status.well_residual > 1e2 * self.nonlinear_solver.spec.tolerance
        )
        converged = self.physics.engine.post_newtonloop(dt, t, converged)
        solver.stats.update(converged, status)

        self.time.append(t)
        self.n_newton_iters.append(status.n_newton)
        self.time_step_size.append(dt)

        self.timer.node["simulation"].stop()

        return converged

    def set_well_rhs(self, Dt, inj_rate, event1, event2):
        if self.physics.engine.t >= 25 * Dt and self.physics.engine.t < 50 * Dt and event1:
            print('At 25 years, start injecting in the second well')
            self.inj_rate = [inj_rate, inj_rate]
            #self.inj_rate = [inj_rate, self.zero]
            event1 = False
        elif self.physics.engine.t >= 50 * Dt and event2:
            print('At 50 years, stop injection for both wells')
            self.inj_rate = [self.zero, self.zero]
            self.specs['check_rates'] = False  # after injection stop checking rates
            event2 = False

        return event1, event2

    def set_well_rates(self, Dt, inj_rate, event1, event2):
        if self.physics.engine.t >= 25 * Dt and self.physics.engine.t < 50 * Dt and event1:
            print('At 25 years, start injecting in the second well')
            self.inj_rate = [inj_rate, inj_rate]
            self.physics.set_well_controls(wctrl=self.reservoir.wells[1].control,
                                            control_type=well_control_iface.MASS_RATE,
                                            is_inj=True,
                                            target=self.inj_rate[1],
                                            phase_name='V',
                                            inj_composition=self.inj_stream[:-1],
                                            inj_temp=self.inj_stream[-1]
                                        )
            event1 = False

        elif self.physics.engine.t >= 50 * Dt and event2:
            print('At 50 years, stop injection in both wells')
            self.inj_rate = [self.zero, self.zero]
            for i in range(2):
                self.physics.set_well_controls(wctrl=self.reservoir.wells[i].control,
                                            control_type=well_control_iface.MASS_RATE,
                                            is_inj=True,
                                            target=self.inj_rate[i],
                                            phase_name='V',
                                            inj_composition=self.inj_stream[:-1],
                                            inj_temp=self.inj_stream[-1]
                                            )
            self.specs['check_rates'] = False  # after injection stop checking rates
            event2 = False

        return event1, event2

    def a_quiver_plot(self, ids_cells, centroids):
        # Get coordinates of start and end points
        start = centroids[ids_cells[:, 0]]  # shape (n_edges, 3)
        end = centroids[ids_cells[:, 1]]

        # Choose 2D plane: x-z
        x_start, z_start = start[:, 0], start[:, 2]
        x_dir = end[:, 0] - start[:, 0]
        z_dir = end[:, 2] - start[:, 2]

        # Plot the quiver plot
        plt.figure(figsize=(10, 6))
        plt.quiver(
            x_start, z_start,  # origins
            x_dir, z_dir,  # directions
            angles='xy',
            scale_units='xy',
            scale=1,
            color='blue',
            width=0.0025
        )
        # plt.xlabel("X")
        # plt.ylabel("Z")
        # plt.title("Flow Directions (x–z plane)")
        # # plt.axis("equal")
        # plt.grid(True)
        # plt.show()

    def map_mesh_faces(self):
        """
        Identifies directional face connections (NS, SN, EW, WE) between mesh blocks.
        Groups these face pairs by direction and stores the indices and associated cell pairs.
        Returns those indices in a structured form for further use (e.g., plots).
        """
        nx, ny, nz = self.reservoir.nx, self.reservoir.ny, self.reservoir.nz
        cell_m = np.asarray(self.reservoir.mesh.block_m)
        cell_p = np.asarray(self.reservoir.mesh.block_p)
        cell_ = np.vstack([cell_m.flatten(), cell_p.flatten()]).T
        self.centroids = self.reservoir.discretizer.centroids_all_cells

        # z-direction
        x0 = np.unique(self.centroids[:, 0])

        ids_SN = np.zeros(len(x0) * nz, dtype=np.int32)
        ids_cells_SN = np.zeros((len(x0) * nz, 2), dtype=np.int32)

        ids_NS = np.zeros(len(x0) * nz, dtype=np.int32)
        ids_cells_NS = np.zeros((len(x0) * nz, 2), dtype=np.int32)

        for i, x in enumerate(x0):
            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 0] - x) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 0] - x) < 1.e-6),
                self.centroids[cell_m, 2] < self.centroids[cell_p, 2]))[0]
            ids_SN[i * (nz - 1):(i + 1) * (nz - 1)] = ids
            ids_cells_SN[i * (nz - 1):(i + 1) * (nz - 1)] = cell_[ids]

            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 0] - x) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 0] - x) < 1.e-6),
                self.centroids[cell_m, 2] > self.centroids[cell_p, 2]))[0]
            ids_NS[i * (nz - 1):(i + 1) * (nz - 1)] = ids
            ids_cells_NS[i * (nz - 1):(i + 1) * (nz - 1)] = cell_[ids]
        # a_quiver_plot(ids_cells_NS, self.centroids); plt.title('NS'); plt.show()
        # a_quiver_plot(ids_cells_SN, self.centroids); plt.title('SN'); plt.show()

        # x-direction
        z0 = np.unique(self.centroids[:, 2])

        ids_EW = np.zeros(len(z0) * nx, dtype=np.int32)
        ids_cells_EW = np.zeros((len(z0) * nx, 2), dtype=np.int32)

        ids_WE = np.zeros(len(z0) * nx, dtype=np.int32)
        ids_cells_WE = np.zeros((len(z0) * nx, 2), dtype=np.int32)

        for i, z in enumerate(z0):
            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 2] - z) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 2] - z) < 1.e-6),
                self.centroids[cell_m, 0] > self.centroids[cell_p, 0]))[0]
            ids_EW[i * (nx - 1):(i + 1) * (nx - 1)] = ids
            ids_cells_EW[i * (nx - 1):(i + 1) * (nx - 1)] = cell_[ids]

            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 2] - z) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 2] - z) < 1.e-6),
                self.centroids[cell_m, 0] < self.centroids[cell_p, 0]))[0]
            ids_WE[i * (nx - 1):(i + 1) * (nx - 1)] = ids
            ids_cells_WE[i * (nx - 1):(i + 1) * (nx - 1)] = cell_[ids]
        # a_quiver_plot(ids_cells_EW, self.centroids); plt.title('EW'); plt.show()
        # a_quiver_plot(ids_cells_WE, self.centroids); plt.title('WE'); plt.show()

        self.ids_list = {'NS': ids_NS, 'SN': ids_SN, 'EW': ids_EW, 'WE': ids_WE}
        self.ids_cells_list = {'NS': ids_cells_NS, 'SN': ids_cells_SN, 'EW': ids_cells_EW, 'WE': ids_cells_WE}

        return

    def plot_fluxes(self, property_array, time_vector, ts):
        nx, ny, nz = self.reservoir.nx, self.reservoir.ny, self.reservoir.nz
        figure_folder = os.path.join(self.output_folder, 'figures', 'fluxes')
        os.makedirs(figure_folder, exist_ok=True)

        for id_key in ['SN']:
            for phase_idx, phase_name in enumerate(self.physics.phases):
                for comp_idx, comp_name in enumerate(self.components):
                    face_centroids_x = (self.centroids[self.ids_cells_list[id_key][:, 0], 0] + self.centroids[
                        self.ids_cells_list[id_key][:, 1], 0]) / 2
                    face_centroids_z = (self.centroids[self.ids_cells_list[id_key][:, 0], 2] + self.centroids[
                        self.ids_cells_list[id_key][:, 1], 2]) / 2

                    # Determine shape and indexing logic
                    if id_key in {'SN', 'NS'}:
                        shape = (nz - 1, nx)
                        index_fn = lambda i, j: i + j * (nz - 1)
                    elif id_key in {'EW', 'WE'}:
                        shape = (nz, nx - 1)
                        index_fn = lambda i, j: j + i * (nx - 1)
                    else:
                        raise ValueError(f"Unsupported id_key: {id_key}")

                    temp_x, temp_z = np.zeros(shape), np.zeros(shape)
                    diff_flux = np.zeros(shape)
                    darcy_flux = np.zeros(shape)
                    disp_flux = np.zeros(shape)

                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            idx = index_fn(i, j)
                            temp_x[i, j] = face_centroids_x[idx]
                            temp_z[i, j] = face_centroids_z[idx]
                            diff_flux[i, j] = property_array[f'diff_fluxes_{phase_name}_{comp_name}_{id_key}'][0, idx]
                            darcy_flux[i, j] = property_array[f'darcy_fluxes_{phase_name}_{comp_name}_{id_key}'][0, idx]
                            disp_flux[i, j] = property_array[f'disp_fluxes_{phase_name}_{comp_name}_{id_key}'][0, idx]

                            xi = property_array[f'x_{phase_name}_{comp_name}']

                    plt.figure(dpi = 100, figsize=(8, 8))
                    plt.suptitle(f"{comp_name}, {phase_name}, in the {id_key} direction @ {time_vector[0]} days")
                    plt.subplot(5, 1, 1)
                    c = plt.pcolor(self.centroids[:, 0].reshape(nz, nx), self.centroids[:, 2].reshape(nz, nx), xi.reshape(nz, nx), cmap='coolwarm')
                    plt.colorbar(c, label = f'x_{phase_name}_{comp_name}')
                    plt.ylabel('z [m]')

                    plt.subplot(5, 1, 2)
                    # pc1 = plt.scatter(face_centroids_x, face_centroids_z, c=property_array[f'diff_fluxes_{phase_name}_{comp_idx}_{id_key}'], s=1, cmap='coolwarm')
                    pc1 = plt.pcolor(temp_x, temp_z, diff_flux, vmin = -np.max(np.abs(diff_flux)), vmax = np.max(np.abs(diff_flux)), cmap='coolwarm')
                    plt.colorbar(pc1, aspect = 10, label="Diffusion Flux")
                    plt.ylim(0, 1200); plt.xlim(0, 8400)
                    plt.ylabel('z [m]')

                    plt.subplot(5, 1, 3)
                    # pc2 = plt.scatter(self.centroids[:, 0], self.centroids[:, 2], c = property_array[f'vel_{phase_name}'], s=1, cmap='coolwarm')
                    velocity = property_array[f'vel_{phase_name}'][0, :].reshape(nz, nx)
                    pc2 = plt.pcolor(self.centroids[:, 0].reshape(nz, nx), self.centroids[:, 2].reshape(nz, nx), velocity, cmap='coolwarm')
                    plt.colorbar(pc2, aspect = 10, label="Velocities")
                    plt.ylim(0, 1200); plt.xlim(0, 8400)
                    plt.ylabel('z [m]')

                    plt.subplot(5, 1, 4)
                    # pc3 = plt.scatter(face_centroids_x, face_centroids_z, c = property_array[f'disp_fluxes_{phase_name}_{comp_idx}_{id_key}'], s=1, cmap='coolwarm')
                    pc3 = plt.pcolor(temp_x, temp_z, disp_flux, vmin = -np.max(np.abs(disp_flux)), vmax = np.max(np.abs(disp_flux)), cmap='coolwarm')
                    plt.colorbar(pc3, aspect = 10, label="Disp Flux")
                    plt.ylim(0, 1200); plt.xlim(0, 8400)
                    plt.ylabel('z [m]')

                    plt.subplot(5, 1, 5)
                    # pc4 = plt.scatter(face_centroids_x, face_centroids_z, c = property_array[f'darcy_fluxes_{phase_name}_{comp_idx}_{id_key}'] , s=1, cmap='coolwarm')
                    pc4 = plt.pcolor(temp_x, temp_z, darcy_flux, vmin = -np.max(np.abs(darcy_flux)), vmax = np.max(np.abs(darcy_flux)), cmap='coolwarm')
                    plt.colorbar(pc4, aspect = 10, label="Darcy Flux")
                    plt.ylim(0, 1200); plt.xlim(0, 8400)
                    plt.xlabel('x [m]')
                    plt.ylabel('z [m]')

                    plt.tight_layout()
                    plt.savefig(os.path.join(figure_folder, f'fluxes_{id_key}_{phase_name}_{comp_name}_at_ts_{ts}.png'))
                    plt.close()

    def plot_properties(self, property_array, time_vector, ts):
        for i, name in enumerate(property_array.keys()):
            if 'fluxes' not in name:
                plt.figure(figsize=(10, 2))
                plt.title(f'{name} @ year {time_vector[0] / 365}')
                c = plt.pcolor(self.grid[0], self.grid[1], property_array[name][0].reshape(self.nz, self.nx), cmap='cividis')
                try:
                    plt.colorbar(c, aspect=10, label=self.output.variable_units[name])
                except:
                    plt.colorbar(c, aspect=10)
                plt.xlabel('x [m]');
                plt.ylabel('z [m]')
                fig_dir = os.path.join(self.output_folder, 'figures', f'{name}')
                os.makedirs(fig_dir, exist_ok=True)
                fig_path = os.path.join(fig_dir, f'{name}_ts_{ts}.png')
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close()
        return

    def plot_reservoir(self):
        nx = self.reservoir.nx
        nz = self.reservoir.nz
        nb = self.reservoir.n
        n_vars = self.physics.n_vars
        vars = self.physics.vars
        self.grid = np.meshgrid(np.linspace(0, 8400, nx), np.linspace(0, 1200, nz))
        poro = self.reservoir.global_data['poro']
        op_num = np.array(self.reservoir.mesh.op_num)[:self.reservoir.n] + 1

        plt.figure(dpi=100, figsize=(10, 2))
        plt.title('Facies')
        c = plt.pcolor(self.grid[0], self.grid[1], op_num.reshape(nz, nx), cmap='jet', vmin=min(op_num), vmax=max(op_num))
        plt.colorbar(c, ticks=np.arange(1, 8))
        plt.xlabel('x [m]');
        plt.ylabel('z [m]')
        # centroids = m.reservoir.discretizer.centroids_all_cells
        centroids = self.reservoir.centroids
        plt.scatter(centroids[self.reservoir.well_cells[0], 0], centroids[self.reservoir.well_cells[0], 2], marker='x', c='r', s=5)
        plt.scatter(centroids[self.reservoir.well_cells[1], 0], centroids[self.reservoir.well_cells[1], 2], marker='x', c='r', s=5)
        plt.savefig(os.path.join(self.output_folder, f'op_num.png'), bbox_inches='tight')
        plt.close()

        solution_vector = np.array(self.physics.engine.X)
        for i, name in enumerate(vars):
            plt.figure(figsize=(10, 2))
            plt.title(name)
            c = plt.pcolor(self.grid[0], self.grid[1], np.round(solution_vector[i::n_vars][:nb], 2).reshape(nz, nx), cmap='jet')
            plt.colorbar(c, aspect=10)
            plt.xlabel('x [m]');
            plt.ylabel('z [m]')
            plt.savefig(os.path.join(self.output_folder, f'initial_conditions_{name}.png'), bbox_inches='tight')
            plt.close()

    def print_darts(self):
        print(r"""
        ------------------------------------------------------
         _____                 _____     _______     _____
        |  __ \       /\      |  __ \   |__   __|  /  ____|
        | |  | |     /  \     | |__) |     | |     | (___
        | |  | |    / /\ \    |  _  /      | |      \___ \
        | |__| |   / ____ \   | | \ \      | |      ____) |
        |_____/   /_/    \_\  |_|  \_\     |_|     |_____/
        ------------------------------------------------------
        """)


class ModBrooksCorey:
    def __init__(self, corey, phase):

        self.phase = phase

        if self.phase == "Aq":
            self.k_rw_e = corey.krwe
            self.swc = corey.swc
            self.sgc = 0
            self.nw = corey.nw
        else:
            self.k_rg_e = corey.krge
            self.sgc = corey.sgc
            self.swc = 0
            self.ng = corey.ng

    def evaluate(self, sat):
        if self.phase == "Aq":
            Se = (sat - self.swc)/(1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rw_e * Se ** self.nw
        else:
            Se = (sat - self.sgc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rg_e * Se ** self.ng

        return k_r


class ModCapillaryPressure:
    def __init__(self, corey):
        self.swc = corey.swc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        # self.labda = 3
        self.eps = 1e-10
        self.pcmax = corey.pcmax
        self.c2 = corey.c2

    def evaluate(self, sat):
        sat_w = sat[1]
        # sat_w = sat
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        # pc = self.p_entry * self.eps ** (1/self.labda) * Se ** (-1/self.labda)  # for p_entry to non-wetting phase
        pc_b = self.p_entry * Se ** (-1/self.c2) # basic capillary pressure
        pc = self.pcmax * erf((pc_b * np.sqrt(np.pi)) / (self.pcmax * 2)) # smoothened capillary pressure
        # if Se > 1 - self.eps:
        #     pc = 0

        # pc = self.p_entry
        Pc = np.array([0, pc], dtype=object)  # V, Aq
        return Pc


class BrooksCorey:
    def __init__(self, wetting: bool):
        self.sat_wr = 0.15
        # self.sat_nwr = 0.1

        self.lambda_w = 4.2
        self.lambda_nw = 3.7

        self.wetting = wetting

    def evaluate(self, sat_w):
        # From Brooks-Corey (1964)
        Se = (sat_w - self.sat_wr)/(1-self.sat_wr)
        if Se > 1:
            Se = 1
        elif Se < 0:
            Se = 0

        if self.wetting:
            k_r = Se**((2+3*self.lambda_w)/self.lambda_w)
        else:
            k_r = (1-Se)**2 * (1-Se**((2+self.lambda_nw)/self.lambda_nw))

        if k_r > 1:
            k_r = 1
        elif k_r < 0:
            k_r = 0

        return k_r
