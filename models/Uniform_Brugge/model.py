from darts.models.darts_model import DartsModel
from darts.engines import sim_params, ms_well
from darts.nonlinear_solvers import NewtonSolver
import numpy as np

from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from mesh_creator import mesh_creator

from darts.physics.base.physics import PhysicsBase
from darts.physics.blackoil import BlackOilProperties

from darts.physics.properties.black_oil import *


class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        # Solver/time-stepping configuration moved to set_solver() (called at top of reset()).

        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.set_sim_params(first_ts=0.0001, mult_ts=2, max_ts=2, runtime=2000 )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=10)
        self.linear_solver.spec.tolerance = 1e-3
        self.linear_solver.spec.max_iterations = 50

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
            self.reservoir.add_perforation(name, res_cell_idx=idx, well_index=well_index_list[i], well_indexD=0)

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        zero = 1e-12
        epsilon = 1e-13
        phases = ['gas', 'oil', 'wat']
        components = ['g', 'o', 'w']

        self.inj_composition = [1 - 2e-8, 1e-8]
        # initial composition should be backtracked from saturations
        self.ini_stream = [0.001225901537, 0.7711341309]

        pvt = 'Brugge_struct/physics.in'
        property_container = BlackOilProperties(phases_name=phases, components_name=components,
                                                 Mw=np.ones(len(components)), eps_z=epsilon, temperature=1.)

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
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        nc = len(components)
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[0.399] + [2e-3] * (nc - 1),  # p [bar], z (3 components → 2 z axes)
                                     axes_origin=[1.0] + [epsilon] * (nc - 1),
                                     epsilon_z=epsilon, extrapolation_flag=True)
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
