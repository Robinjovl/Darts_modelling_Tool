from darts.models.thmc_model import THMCModel
from reservoir import UnstructReservoirCustom
from darts.physics.mech.poroelasticity import Poroelasticity
from darts.engines import value_vector, sim_params
from darts.tools.keyword_file_tools import load_single_keyword

import numpy as np
import os

from darts.physics.super.property_container import PropertyContainer
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
        super().__init__()

    def set_solver_params(self):
        super().set_solver_params()
        self.params.linear_type = sim_params.cpu_gmres_fs_cpr
        #self.params.linear_type = sim_params.cpu_superlu
        self.params.first_ts = 0.0001
        self.params.mult_ts = 2
        self.params.max_ts = 5
        self.params.tolerance_newton = 1e-6
        self.params.tolerance_linear = 1e-8
        self.params.max_i_newton = 20

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
        self.idata.rock.conductivity = 0.836 * 86400.0 / 1000  # [kJ/m/day/K]
        self.idata.rock.heat_capacity = 167.2 * 1000.0  # [kJ/m3/K]
        self.idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update

        # TODO: Only for a single-phase physics
        self.idata.fluid.Mw = 18.015
        self.idata.fluid.compressibility = 1.45e-5
        self.idata.fluid.viscosity = 1.0
        self.idata.fluid.density = 666.854632

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

        self.idata.obl.n_points = 400
        self.idata.obl.zero = 1e-9
        self.idata.obl.min_p = 0.0
        self.idata.obl.max_p = 1000.
        self.idata.obl.min_t = 273.15 + 20
        self.idata.obl.max_t = 273.15 + 200
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
        super().set_input_data()

    def set_physics(self):
        p_ref = 350.0
        t_ref = 300.0

        if self.physics_type == 'single_phase':
            Mw = [self.idata.fluid.Mw]
            components = ['H2O']
            phases = ['wat']
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, min_z=self.idata.obl.min_z, temperature=t_ref)

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
                                                   Mw=Mw, min_z=self.idata.obl.min_z)

            """ properties correlations """
            property_container.flash_ev = SinglePhase(nc=1)
            property_container.density_ev = dict([('wat', DensityBasic(compr=self.idata.fluid.compressibility,
                                                                       dens0=self.idata.fluid.density,
                                                                       p0=p_ref))])
            property_container.viscosity_ev = dict([('wat', ConstFunc(self.idata.fluid.viscosity))])

            property_container.rel_perm_ev = dict([('wat', ConstFunc(1.0))])
            # rock compressibility is treated inside engine
            property_container.rock_compr_ev = ConstFunc(1.0)

            property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=self.idata.rock.heat_capacity, tref=t_ref))])
            property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0, tref=t_ref)  #TODO use hcap from idata? see https://gitlab.com/open-darts/open-darts/-/issues/19
            property_container.conductivity_ev = dict([('wat', ConstFunc(1.0))])
        elif self.physics_type == 'dead_oil' or self.physics_type == 'dead_oil_thermal':
            components = ['w', 'o']
            phases = ['wat', 'oil']
            self.cell_property = ['pressure'] + ['water']

            property_container = ModelProperties(phases_name=phases, components_name=components, min_z=self.idata.obl.min_z)

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
            property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        property_container.rock_density_ev = ConstFunc(self.idata.rock.density)
        # create physics
        state_spec = Poroelasticity.StateSpecification.PT if self.thermal else Poroelasticity.StateSpecification.P
        self.physics = Poroelasticity(components, phases, self.timer, state_spec=state_spec, n_points=self.idata.obl.n_points,
                                      min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p,
                                      min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                      min_t=self.idata.obl.min_t, max_t=self.idata.obl.max_t,
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

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=None)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        if self.thermal:
            self.temperature = vec_state_as_np[-1]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0, 1], dtype=np.intp)

        for j in self.ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        mass_source = np.zeros(self.nc)

        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
