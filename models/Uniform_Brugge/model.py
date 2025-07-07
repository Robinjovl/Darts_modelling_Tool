from darts.models.cicd_model import CICDModel
from darts.engines import sim_params
import numpy as np

from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from mesh_creator import mesh_creator

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.black_oil import *


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=0.0001, mult_ts=2, max_ts=2, runtime=2000,
                            tol_newton=1e-3, tol_linear=1e-3, it_newton=10, it_linear=50)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        """Reservoir"""
        # GMSH file where mesh will be saved
        mesh_file = 'Brugge_model.msh'

        # description of structured Brugge model
        (nx, ny, nz) = (139, 48, 9)
        struct_mesh_path = 'Brugge_struct/dxdydz.in'
        ACTNUM_path = 'Brugge_struct/ACTNUM.in'
        depth_path = 'Brugge_struct/depth.in'
        well_coord_path = 'Brugge_struct/well_coord_Brugge.txt'
        lc_bound = [200, 2500]
        thickness = 72  # the thickness of real reservoir, in meter
        random_seed = 999  # set different seed to generate different versions of mesh with the same set of parameters

        mesh_creator(random_seed, nx, ny, nz, lc_bound, thickness, struct_mesh_path, ACTNUM_path,
                     depth_path, mesh_file, well_coord_path)

        # Some permeability input data for the simulation
        const_perm = 500
        permx = const_perm  # Matrix permeability in the x-direction [mD]
        permy = const_perm  # Matrix permeability in the y-direction [mD]
        permz = const_perm  # Matrix permeability in the z-direction [mD]
        poro = 0.2  # Matrix porosity [-]
        frac_aper = 1e-4  # Aperture of fracture cells (but also takes a list of apertures for each segment) [m]

        # Instance of unstructured reservoir class from reservoir.py file.
        # When calling this class constructor, the def __init__(self, arg**) is executed which created the instance of
        # the class and constructs the object. In the process, the mesh is loaded, mesh information is calculated and
        # the discretization is executed. Besides that, also the boundary conditions of the simulations are
        # defined in this class --> in this case constant pressure/rate at the left (x==x_min) and right (x==x_max) side
        self.reservoir = UnstructReservoir(timer=self.timer, mesh_file=mesh_file, permx=permx, permy=permy, permz=permz,
                                           poro=poro, frac_aper=frac_aper)
        self.reservoir.physical_tags['matrix'] = [999]

        return

    def set_wells(self):
        well_coord = np.genfromtxt('Brugge_struct/well_coord_Brugge.txt')
        n_injector = 10  # the first 10 wells are injectors
        n_wells = 30  # the number of wells
        calc_equiv_WI = True
        if calc_equiv_WI:
            well_index_list = [None] * len(well_coord)
        else:
            well_index_list = [296.65303668, 69.71905642, 27.14929434, 27.58575654, 53.01869826,
                               135.80457602, 345.34715322, 80.69146768, 74.07293499, 243.34286931,
                               482.48726884, 592.37938461, 307.72095815, 542.44279506, 63.58456305,
                               499.53586907, 213.09805386, 303.97957677, 78.80966839, 791.4220401,
                               829.44984229, 794.13401768, 761.08006179, 62.37906275, 616.74501491,
                               475.63754963, 397.50862698, 478.21742722, 504.06328513, 655.32259614]

        for i, wc in enumerate(well_coord):
            if i < n_injector:
                name = "I" + str(i + 1)
            else:
                name = "P" + str(i + 1 - n_injector)

            self.reservoir.add_well(name)
            idx = self.reservoir.find_cell_index(wc)
            self.reservoir.add_perforation(name, cell_index=idx, well_index=well_index_list[i], well_indexD=0)

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        zero = 1e-12
        phases = ['gas', 'oil', 'wat']
        components = ['g', 'o', 'w']

        self.inj_composition = [1 - 2e-8, 1e-8]
        # initial composition should be backtracked from saturations
        self.ini_stream = [0.001225901537, 0.7711341309]

        pvt = 'Brugge_struct/physics.in'
        property_container = ModelProperties(phases_name=phases, components_name=components, pvt=pvt, min_z=zero/10)

        """ properties correlations """
        property_container.flash_ev = flash_black_oil(pvt)
        property_container.density_ev = dict([('gas', DensityGas(pvt)),
                                              ('oil', DensityOil(pvt)),
                                              ('wat', DensityWat(pvt))])
        property_container.viscosity_ev = dict([('gas', ViscGas(pvt)),
                                                ('oil', ViscOil(pvt)),
                                                ('wat', ViscWat(pvt))])
        property_container.rel_perm_ev = dict([('gas', GasRelPerm(pvt)),
                                               ('oil', OilRelPerm(pvt)),
                                               ('wat', WatRelPerm(pvt))])
        property_container.capillary_pressure_ev = dict([('pcow', CapillaryPressurePcow(pvt)),
                                                         ('pcgo', CapillaryPressurePcgo(pvt))])

        property_container.rock_compress_ev = RockCompactionEvaluator(pvt)

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=500, min_p=1, max_p=200, min_z=zero / 10, max_z=1 - zero / 10)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 170.,
                              self.physics.vars[1]: self.ini_stream[0],
                              self.physics.vars[2]: self.ini_stream[1],
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=180., inj_composition=self.inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=150.)

    def run_custom(self, export_to_vtk=False):
        if export_to_vtk:
            X = np.array(self.engine.X, copy=False)
            for ith_prop in range(self.physics.n_vars):
                self.property_array[ith_prop, :] = X[ith_prop::self.physics.n_vars]

        time_step = self.report_step
        even_end = int(self.T / time_step) * time_step
        time_step_arr = np.ones(int(self.T / time_step)) * time_step
        if self.T - even_end > 0:
            time_step_arr = np.append(time_step_arr, self.T - even_end)

        from darts.engines import well_control_iface
        for ith_step, ts in enumerate(time_step_arr):
            # print("Running time: %d days" % ts)
            for i, w in enumerate(self.reservoir.wells):
                if 'I' in w.name:
                    self.physics.set_well_controls(well=w, is_control=True, control_type=well_control_iface.BHP,
                                                   is_inj=True, target=175., inj_composition=self.inj_composition)
                else:
                    self.physics.set_well_controls(well=w, is_control=True, control_type=well_control_iface.BHP,
                                                   is_inj=False, target=125.)
                    # w.control = self.physics.new_rate_water_prod(self.inj_prod_rate)

            self.engine.run(ts)
            self.engine.report()

            if export_to_vtk:
                X = np.array(self.engine.X, copy=False)
                for ith_prop in range(self.physics.n_vars):
                    self.property_array[ith_prop, :] = X[ith_prop::self.physics.n_vars]

                self.property_array[-1, :] = _Backward1_T_Ph_vec(X[0::self.physics.n_vars] / 10,
                                                                 X[1::self.physics.n_vars] / 18.015)  # calc temperature
                self.reservoir.unstr_discr.write_to_vtk('vtk_data', self.property_array,
                                                        ['pressure', 'enthalpy', 'temperature'], ith_step + 1)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, pvt, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)

        super().__init__(phases_name, components_name, Mw, min_z=min_z, temperature=1.)
        self.pvt = pvt
        self.surf_dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.surf_oil_dens = self.surf_dens[0]
        self.surf_wat_dens = self.surf_dens[1]
        self.surf_gas_dens = self.surf_dens[2]

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

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        (xgo, V, pbub) = self.flash_ev.evaluate(pressure, zc)
        for i in range(self.nph):
            self.x[i, i] = 1

        if V < 0:
            self.ph = np.array([1, 2], dtype=np.intp)
        else:  # assume oil and water are always exists
            self.x[1][0] = xgo
            self.x[1][1] = 1 - xgo
            self.ph = np.array([0, 1, 2], dtype=np.intp)

        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, pbub, xgo)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, pbub)  # output in [cp]

        self.nu[2] = zc[2]
        # two phase undersaturated condition
        if pressure > pbub:
            self.nu[0] = 0
            self.nu[1] = zc[1]
        else:
            self.nu[1] = zc[1] / (1 - xgo)
            self.nu[0] = 1 - self.nu[1] - self.nu[2]

        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0], self.sat[2])

        pcow = self.capillary_pressure_ev['pcow'].evaluate(self.sat[2])
        pcgo = self.capillary_pressure_ev['pcgo'].evaluate(self.sat[0])

        self.pc = np.array([-pcgo, 0, pcow])

        return

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.ph = []
        for j in range(self.nph):
            if zc[j] > self.min_z:
                self.ph.append(j)
            self.dens_m[j] = self.density_ev[self.phases_name[j]].dens_sc

        self.ph = np.array(self.ph)
        self.nu = zc
        self.compute_saturation(self.ph)

        return self.sat, self.dens_m
