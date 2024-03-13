from darts.models.thmc_model import THMCModel
from reservoir import UnstructReservoirCustom
from darts.physics.mech.poroelasticity import Poroelasticity
from darts.engines import value_vector, sim_params
from darts.tools.keyword_file_tools import load_single_keyword

import numpy as np

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
        self.model_folder = model_folder
        self.uniform_props = uniform_props
        self.physics_type = physics_type
        self.discretizer_name = 'mech_discretizer'

        # call base class constructor
        super().__init__()

    def set_solver_params(self):
        super().set_solver_params()
        self.params.linear_type = sim_params.cpu_superlu # cpu_gmres_fs_cpr # cpu_superlu
        self.params.first_ts = 0.0001
        self.params.mult_ts = 2
        self.params.max_ts = 5
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 12

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
            permeability = 100.0 # [mD]
            E = 1 # [10 GPa]
            nu = 0.2
            p_init = 1000 # [bar]
        else:
            porosity = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/poro.txt', 'PORO', cache=0).
                                        reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()
            permeability = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/perm.txt', 'PERM', cache=0).
                                        reshape(self.nz, self.ny, self.nx, 3), 0, 2), axis=2).flatten()
            E = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/young.txt', 'YOUNG', cache=0).
                                    reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()
            nu = 0.2
            p_init = np.flip(np.swapaxes(load_single_keyword(self.model_folder + '/ref_pres.txt', 'REF_PRESSURE', cache=0).
                                    reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()

        self.idata = InputData(type_hydr='isothermal', type_mech='poroelasticity')
        self.idata.rock.heat_capacity = 167.2 * 1000.0 # [kJ/m3/K]
        self.idata.rock.conductivity = 181.44  # [kJ/m/day/K]  #TODO why it was not there before
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

        # TODO: Only for a single-phase physics
        self.idata.fluid.Mw = 18.015
        self.idata.fluid.compressibility = 1.e-5
        self.idata.fluid.viscosity = 1.0
        self.idata.fluid.density = 1014.0

        self.idata.initial.initial_temperature = 273.15 + 50  # [K]
        self.idata.initial.initial_pressure = p_init  # [bar]
        self.idata.initial.initial_displacements = [0., 0., 0.]  # [m]
        if self.physics_type == 'dead_oil':
            self.idata.initial.initial_composition = [0.67]

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
        if self.physics_type == 'single_phase':
            Mw = [self.idata.fluid.Mw]
            components = ['H2O']
            phases = ['wat']
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, min_z=self.idata.obl.min_z, temperature=273.15 + 50)

            """ properties correlations """
            property_container.flash_ev = SinglePhase(nc=1)
            property_container.density_ev = dict([('wat', DensityBasic(compr=self.idata.fluid.compressibility,
                                                                       dens0=self.idata.fluid.density))])
            property_container.viscosity_ev = dict([('wat', ConstFunc(self.idata.fluid.viscosity))])

            property_container.rel_perm_ev = dict([('wat', ConstFunc(1.0))])
            # rock compressibility is treated inside engine
            property_container.rock_compr_ev = ConstFunc(1.0)
        elif self.physics_type == 'dead_oil':
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
        self.physics = Poroelasticity(components, phases, self.timer, n_points=self.idata.obl.n_points,
                                      min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p,
                                      min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                      discretizer=self.discretizer_name)
        self.physics.add_property_region(property_container)

        self.engine = self.physics.init_physics(discretizer=self.discretizer_name, platform='cpu')
        return

    def set_initial_conditions(self):
        if self.reservoir.thermoporoelasticity:
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                           initial_pressure=self.reservoir.p_init,
                                                           initial_composition=self.reservoir.z_init,
                                                           initial_temperature=self.reservoir.t_init,
                                                           initial_displacement=[0.0, 0.0, 0.0])
        else:
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                           initial_pressure=self.reservoir.p_init,
                                                           initial_composition=self.reservoir.z_init,
                                                           initial_displacement=self.reservoir.u_init)
        return 0


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z, temperature=None)

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

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = [0, 1]

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
