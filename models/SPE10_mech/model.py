from darts.models.thmc_model import THMCModel
from reservoir import UnstructReservoirCustom
from darts.physics.mech.poroelasticity import Poroelasticity
from darts.engines import value_vector, sim_params
from darts.tools.keyword_file_tools import load_single_keyword

import numpy as np
import os

from darts.physics.base.property_container import PropertyContainer
from darts.physics.deadoil import DeadOilProperties
from darts.physics.properties.flash import SinglePhase
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.reservoirs.unstruct_reservoir_mech import get_bulk_modulus, get_rock_compressibility, get_isotropic_stiffness
from darts.reservoirs.unstruct_reservoir_mech import get_biot_modulus
from darts.input.input_data import InputData
from reservoir import UnstructReservoirCustom

class Model(THMCModel):
    def __init__(self, model_folder, physics_type='dead_oil', uniform_props=False):
        self.model_folder = os.path.join('meshes', model_folder)
        self.uniform_props = uniform_props
        self.physics_type = physics_type
        self.discretizer_name = 'mech_discretizer'
        if self.physics_type == 'single_phase_thermal' or \
            self.physics_type == 'dead_oil_thermal':
            self.thermal = True
        else:
            self.thermal = False

        # call base class constructor
        # NOTE: solver / time-stepping / Newton / linear-solver configuration moved
        # to set_solver() (called at the top of reset()) per the new convention.
        super().__init__()

    def set_solver(self):
        super().set_solver()

        # Open-source FS-CPR by default -- inject the spec; the engine bypasses
        # sim_params.linear_type. FS-CPR is a PRECONDITIONER (single application),
        # not an outer Krylov loop -- wrap it in GMRES to mirror the proprietary
        # path (bos_gmres + bos_fs_cpr).
        from darts.models.darts_model import DataTS
        from darts.linear_solvers.specs import FSCPRSolverSpec, GMRESSolverSpec
        if not hasattr(self, 'data_ts') or self.data_ts is None:
            self.data_ts = DataTS(self.physics.n_vars)
        mesh = self.reservoir.mesh
        n_blocks = mesh.n_blocks
        n_res_blks = mesh.n_res_blocks
        n_matrix = getattr(self.reservoir, 'n_matrix', n_res_blks)
        n_fracs_mesh = getattr(self.reservoir, 'n_fracs', 0)
        # Match proprietary engine_pm_cpu.cpp:136 convention:
        #   n_res  = n_matrix + n_fracs  (matrix + fracture cells treated as "reservoir")
        #   n_fracs= 0   (zero gap-DOF rows -- FS_UPG not yet supported)
        #   n_wells= n_blocks - n_res_blocks
        fs_cpr = FSCPRSolverSpec(
            force_amg_asymmetric=True,
            n_res=n_matrix + n_fracs_mesh,
            n_fracs=0,
            n_wells=n_blocks - n_res_blks,
        )
        # Single solver declaration: the FS-CPR spec drives _apply_solver on the
        # open-source CPU build; on the proprietary build _apply_solver applies
        # proprietary_linear_type (bos_fs_cpr) to params.linear_type. No model-level
        # params.linear_type needed -- its open-source value was the engine default
        # (cpu_superlu) anyway.
        self.linear_solver = GMRESSolverSpec(
            prec=fs_cpr,
            # NOTE: 1e-5 / 50 are the values this model has always effectively run with.
            # Until !280 the engine overwrote a spec's tolerance/max_iterations at init()
            # with sim_params (defaults 1e-5 / 50, globals.h:117), so the spec's numbers were
            # decorative. The spec is authoritative now, so state the values this model has
            # really been running -- keeping behaviour unchanged. FS-CPR does not reach 1e-8 here
            # anyway: asking for it only burns the iteration budget -- 22 of 48 solves exhaust the
            # 200-iteration cap (99 vs 41 linear iterations per Newton).
            tolerance=1e-5,
            max_iterations=50,
            restart=50,
            proprietary_linear_type=sim_params.cpu_gmres_fs_cpr,
        )

        self.data_ts.dt_first = 0.0001
        self.data_ts.dt_mult = 2
        self.data_ts.dt_max = 5
        self.nonlinear_solver.spec.tolerance = 1e-6
        self.params.tolerance_linear = 1e-8
        self.nonlinear_solver.spec.max_iterations = 20

    def set_reservoir(self):
        self.reservoir = UnstructReservoirCustom(timer=self.timer, fluid_vars=self.physics.vars,
                                                 idata=self.idata, model_folder=self.model_folder,
                                                 uniform_props=self.uniform_props)

    def set_input_data(self):
        # figure out nx, ny, nz
        self.nx, self.ny, self.nz = int(self.model_folder.split('_')[-3]), \
                                    int(self.model_folder.split('_')[-2]), \
                                    int(self.model_folder.split('_')[-1])

        # read properties
        if self.uniform_props:
            porosity = 0.375
            permeability = 10.0 # [mD]
            E = 1 # [10 GPa]
            # p_init = 300 * np.ones(self.nx * self.ny * self.nz)  # [bar]
        else:
            porosity = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/poro.txt', 'PORO', cache=0).
                                        reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()
            permeability = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/perm.txt', 'PERM', cache=0).
                                        reshape(self.nz, self.ny, self.nx, 3), 0, 2), axis=2).flatten()
            E = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/young.txt', 'YOUNG', cache=0).
                                    reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()
        p_init = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/ref_pres.txt', 'REF_PRESSURE', cache=0).
                        reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()
        nu = 0.2

        self.idata = InputData(type_hydr='isothermal', type_mech='poroelasticity', init_type = 'gradient')
        self.idata.rock.density = 2650.
        self.idata.rock.porosity = porosity
        self.idata.rock.permx = self.idata.rock.permy = self.idata.rock.permz = permeability
        self.idata.rock.biot = 1.0
        self.idata.rock.E = 1.e+5 * E
        self.idata.rock.nu = nu
        self.idata.rock.compressibility = get_rock_compressibility(
            kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu),
            biot=self.idata.rock.biot, poro0=self.idata.rock.porosity)
        self.idata.rock.stiffness = get_isotropic_stiffness(self.idata.rock.E, self.idata.rock.nu)

        self.idata.rock.th_expn = 9.0 * 1.E-7
        self.idata.rock.th_expn *= get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu)
        self.idata.rock.thermal_conductivity = 0.836 * 86400.0  # [kJ/m/day/K]
        self.idata.rock.heat_capacity = 167.2  # [kJ/m3/K]
        self.idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update

        # TODO: Only for a single-phase physics
        self.idata.fluid.Mw = 18.015
        self.idata.fluid.compressibility = 1.45e-5
        self.idata.fluid.viscosity = 1.0
        self.idata.fluid.density = 666.854632
        self.idata.fluid.heat_capacity = 75. #[kJ/kmol/K]
        self.idata.fluid.thermal_conductivity = 0. # it is not used in the mech. engines

        self.idata.initial.initial_temperature = 273.15 + 50  # [K]
        self.idata.initial.initial_pressure = p_init  # [bar]
        self.idata.initial.initial_displacements = [0., 0., 0.]  # [m]
        if self.physics_type == 'dead_oil' or self.physics_type == 'dead_oil_thermal':
            self.idata.initial.initial_composition = [0.67]

        self.idata.mesh.bnd_tags = {}
        bnd_tags = self.idata.mesh.bnd_tags  # short name
        bnd_tags['BND_X-'] = 991
        bnd_tags['BND_X+'] = 992
        bnd_tags['BND_Y-'] = 993
        bnd_tags['BND_Y+'] = 994
        bnd_tags['BND_Z-'] = 995
        bnd_tags['BND_Z+'] = 996
        self.idata.mesh.matrix_tags = [99991]

        self.idata.obl.zero = 1e-9
        self.idata.obl.p_step = 2.5
        self.idata.obl.p_origin = 0.0
        self.idata.obl.z_step = 2.5e-3
        self.idata.obl.z_origin = self.idata.obl.zero / 10
        self.idata.obl.t_step = 0.45
        self.idata.obl.t_origin = 273.15 + 20
        self.idata.obl.epsilon_z = self.idata.obl.zero/10
        super().set_input_data()

    def set_physics(self):
        p_ref = 350.0
        t_ref = 300.0

        if self.physics_type == 'single_phase':
            Mw = [self.idata.fluid.Mw]
            components = ['H2O']
            phases = ['wat']
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, eps_z=self.idata.obl.epsilon_z, temperature=t_ref)

            """ properties correlations """
            property_container.flash_ev = SinglePhase(nc=1)
            property_container.density_ev = dict([('wat', DensityBasic(compr=self.idata.fluid.compressibility,
                                                                       dens0=self.idata.fluid.density,
                                                                       p0=p_ref))])
            property_container.viscosity_ev = dict([('wat', ConstFunc(self.idata.fluid.viscosity))])

            property_container.rel_perm_ev = dict([('wat', ConstFunc(1.0))])
            # rock compressibility is treated inside engine
            property_container.rock_compr_ev = ConstFunc(1.0)
        elif self.physics_type == 'single_phase_thermal':
            components = ['H2O']
            phases = ['wat']
            Mw = [self.idata.fluid.Mw]

            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, eps_z=self.idata.obl.epsilon_z)

            """ properties correlations """
            property_container.flash_ev = SinglePhase(nc=1)
            property_container.density_ev = dict([('wat', DensityBasic(compr=self.idata.fluid.compressibility,
                                                                       dens0=self.idata.fluid.density,
                                                                       p0=p_ref))])
            property_container.viscosity_ev = dict([('wat', ConstFunc(self.idata.fluid.viscosity))])

            property_container.rel_perm_ev = dict([('wat', ConstFunc(1.0))])
            # rock compressibility is treated inside engine
            property_container.rock_compr_ev = ConstFunc(1.0)

            property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=self.idata.fluid.heat_capacity, tref=t_ref))])
            property_container.conductivity_ev = dict([('wat', ConstFunc(self.idata.fluid.thermal_conductivity))])
        elif self.physics_type == 'dead_oil' or self.physics_type == 'dead_oil_thermal':
            components = ['w', 'o']
            phases = ['wat', 'oil']
            self.cell_property = ['pressure'] + ['water']

            property_container = ModelProperties(phases_name=phases, components_name=components,
                                                 Mw=np.ones(len(phases)), eps_z=self.idata.obl.epsilon_z,
                                                 temperature=None)

            # Define property evaluators based on custom properties
            property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                                  ('oil', DensityBasic(compr=5e-3, dens0=50))])
            property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                    ('oil', ConstFunc(0.03))])
            property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("gas", 0.1, 0.1)),
                                                   ('oil', PhaseRelPerm("oil", 0.1, 0.1))])
            property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=4.18)),
                                                   ('oil', EnthalpyBasic(hcap=0.035))])
            property_container.conductivity_ev = dict([('wat', ConstFunc(1.)),
                                                       ('oil', ConstFunc(1.))])

        property_container.rock_density_ev = ConstFunc(self.idata.rock.density)
        # create physics: [p, z_1, ..., z_{nc-1}, T?]
        state_spec = Poroelasticity.StateSpecification.PT if self.thermal else Poroelasticity.StateSpecification.P
        nz = len(components) - 1
        ax_step = [self.idata.obl.p_step] + [self.idata.obl.z_step] * nz
        ax_origin = [self.idata.obl.p_origin] + [self.idata.obl.z_origin] * nz
        if self.thermal:
            ax_step.append(self.idata.obl.t_step)
            ax_origin.append(self.idata.obl.t_origin)
        self.physics = Poroelasticity(components, phases, self.timer, state_spec=state_spec,
                                      axes_step=ax_step, axes_origin=ax_origin,
                                      epsilon_z=self.idata.obl.epsilon_z,
                                      discretizer=self.discretizer_name)
        self.physics.add_property_region(property_container)

        self.physics.init_physics(discr_type=self.discretizer_name, platform='cpu')

        return

    def set_wells(self):
        centroids = np.array([np.array([c.values[0], c.values[1]]) for
                              c in self.reservoir.discr_mesh.centroids])[:self.reservoir.n_matrix]
        l_min = np.min(self.reservoir.mesh_data.points, axis=0)
        l_max = np.max(self.reservoir.mesh_data.points, axis=0)

        well_coords = np.array([[l_max[0] / 2 - 2, l_max[1] / 2 - 200], [l_max[0] / 2 - 2, l_max[1] / 2 + 200]])
        well_names = ['PRD1', 'INJ1']
        self.well_cell_ids = []
        well_init_depth = l_min[2]
        nodes = np.array(self.reservoir.discr_mesh.nodes)
        elems = np.array(self.reservoir.discr_mesh.elems)
        for i, coord in enumerate(well_coords):
            ids = ((centroids[:, 0] - coord[0]) ** 2 + (centroids[:, 1] - coord[1]) ** 2).argsort()
            self.well_cell_ids.append(ids[:self.nz])
            # adding well
            self.reservoir.add_well(well_names[i], depth=well_init_depth)
            # adding perforations
            for cell_id in ids[:self.nz]:
                cell = elems[cell_id]
                pt_ids = self.reservoir.discr_mesh.elem_nodes[cell.pts_offset:cell.pts_offset + cell.n_pts]
                pts = np.array([nodes[id].values for id in pt_ids])
                # Calculate well_index (very primitive way....):
                rw = 0.1
                dx = np.max(pts, axis=0)[0] - np.min(pts, axis=0)[0]
                dy = np.max(pts, axis=0)[1] - np.min(pts, axis=0)[1]
                dz = np.max(pts, axis=0)[2] - np.min(pts, axis=0)[2]
                perm = np.array(self.reservoir.discr.perms[cell_id].values)
                mean_perm_xx = perm[0]
                mean_perm_yy = perm[4]
                mean_perm_zz = perm[8]
                rp_z = 0.28 * np.sqrt((mean_perm_yy / mean_perm_xx) ** 0.5 * dx ** 2 +
                                      (mean_perm_xx / mean_perm_yy) ** 0.5 * dy ** 2) / \
                       ((mean_perm_xx / mean_perm_yy) ** 0.25 + (mean_perm_yy / mean_perm_xx) ** 0.25)
                wi_x = 0.0
                wi_y = 0.0
                wi_z = 2 * np.pi * np.sqrt(mean_perm_xx * mean_perm_yy) * dz / np.log(rp_z / rw)
                well_index = np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)
                # add perforation
                self.reservoir.add_perforation(self.reservoir.wells[-1], cell_id, well_index=well_index)

    def set_boundary_conditions(self):
        from darts.engines import well_control_iface
        self.physics.set_well_controls(wctrl=self.reservoir.wells[0].control,
                                       control_type=well_control_iface.MOLAR_RATE,
                                       is_inj=False, target=0., phase_name='wat')
        if len(self.reservoir.wells) > 1:
            inj = []
            inj_temp = None
            if self.physics_type == 'single_phase_thermal':
                inj_temp = np.mean(self.reservoir.t_init[self.well_cell_ids[1]])
            elif self.physics_type == 'dead_oil':
                inj = [1.0 - self.idata.obl.zero]
            elif self.physics_type == 'dead_oil_thermal':
                inj = [1.0 - self.idata.obl.zero]
                inj_temp = np.mean(self.reservoir.t_init[self.well_cell_ids[1]])
            self.physics.set_well_controls(wctrl=self.reservoir.wells[1].control,
                                           control_type=well_control_iface.MOLAR_RATE,
                                           is_inj=True, target=0., phase_name='wat', inj_composition=inj, inj_temp=inj_temp)

    def set_boundary_conditions_after_initialization(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            p_cell = self.reservoir.p_init[self.well_cell_ids[i]]
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=np.min(p_cell)-50)
            else:
                inj = []
                inj_temp = None
                if self.physics_type == 'single_phase_thermal':
                    inj_temp = np.mean(self.reservoir.t_init[self.well_cell_ids[1]])
                elif self.physics_type == 'dead_oil':
                    inj = [1.0 - self.idata.obl.zero]
                elif self.physics_type == 'dead_oil_thermal':
                    inj = [1.0 - self.idata.obl.zero]
                    inj_temp = np.mean(self.reservoir.t_init[self.well_cell_ids[1]]) - 25
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=np.max(p_cell) + 50., inj_composition=inj,
                                               inj_temp=inj_temp)
        return 0

    def set_initial_conditions(self):
        input_distribution = {'pressure': self.reservoir.p_init}
        input_distribution.update({comp: self.reservoir.z_init[i] for i, comp in enumerate(self.physics.components[:-1])})
        if self.reservoir.thermoporoelasticity:
            input_distribution['temperature'] = self.reservoir.t_init
            input_displacement = [0.0, 0.0, 0.0]
        else:
            input_displacement = self.reservoir.u_init

        self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                       input_distribution=input_distribution,
                                                       input_displacement=input_displacement)
        return 0

class ModelProperties(DeadOilProperties):
    def evaluate(self, state):
        super().evaluate(state)
        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, self.mass_source
