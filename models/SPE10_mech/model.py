from scipy.interpolate import interp1d
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
from darts.engines import well_control_iface
from darts.models.thmc_model import THMCModel
from darts.physics.mech.poroelasticity import Poroelasticity
from darts.engines import value_vector, sim_params
from darts.tools.keyword_file_tools import load_single_keyword
from darts.physics.super.initialize import Initialize

from reservoir import UnstructReservoirCustom

def fmt_e(x : float):
    return "{:.3e}".format(x) if np.isscalar(x) else str(x)

def fmt(x : float):
    return "{:.3}".format(x) if np.isscalar(x) else str(x)

class Model(THMCModel):
    def __init__(self, model_folder, physics_type='dead_oil', 
                 uniform_props=False, wells_type=None, 
                 decouple_geomech=False, generate_mesh=False, dummy='no'):
        self.model_folder = os.path.join('meshes', model_folder)
        self.uniform_props = uniform_props
        self.physics_type = physics_type
        self.discretizer_name = 'mech_discretizer'
        if self.physics_type == 'single_phase_thermal' or \
            self.physics_type == 'dead_oil_thermal':
            self.thermal = True
        else:
            self.thermal = False
        self.decouple_geomech = decouple_geomech
        self.generate_mesh = generate_mesh
        self.wells_type = wells_type
        
        if dummy == 'yes':  # save time for proxy run
            return
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
                                                 uniform_props=self.uniform_props, generate_mesh=self.generate_mesh)

    def set_input_data(self):
        # figure out nx, ny, nz
        dims=os.path.basename(self.model_folder).split('_')
        nx, ny, nz = int(dims[-3]), int(dims[-2]), int(dims[-1])

        # set properties
        porosity =  0.1
        #permeability = 1000 # [mD] # this matched thm and analytical solution
        permeability = 10 # [mD] # this matches proxy and thm
        
        E = 12 # Young modulus [GPa]
        #E = 22  # GPa, Dinantian carbonate 
        #E = 12  # GPa, Indiana Limestone 
        
        p_init = 300 * np.ones(nx * ny * nz)  # [bar]

        if 'thermal' in self.physics_type:
            self.idata = InputData(type_hydr='thermal', type_mech='thermoporoelasticity', init_type = 'gradient')
        else:
            self.idata = InputData(type_hydr='isothermal', type_mech='poroelasticity', init_type = 'gradient')

        self.idata.other.nx, self.idata.other.ny, self.idata.other.nz = nx, ny, nz

        self.idata.other.perm_frac = False
        if self.idata.other.perm_frac:
            # conductive fracture
            porosity = 1.0
            permeability = 1e6 # [mD]

        self.idata.rock.density = 2650. # kg/m63
        self.idata.rock.porosity = porosity
        self.idata.rock.permx = self.idata.rock.permy = self.idata.rock.permz = permeability
        #self.idata.rock.biot = 1  # rock compressibility will be 0
        self.idata.rock.biot = 0.7  # 0.8 to match dp with geos
        self.idata.rock.E = 1.e+4 * E  # convert units to bars
        self.idata.rock.nu = 0.25  # poisson ratio

        # define permeable reservoir geometric boundaries
        self.idata.other.rsv_top = 2000  # [m]
        self.idata.other.rsv_bottom = 2400# [m]
        
        # lateral reservoir boundaries
        self.idata.other.rsv_xy = 1000.   # m, laterally limited (rsv width will be self.rsv_xy*2)
        #self.idata.other.rsv_xy = 1e5  # m, "infinite" laterally
        
        self.idata.other.rsv_x1 = -self.idata.other.rsv_xy
        self.idata.other.rsv_x2 = self.idata.other.rsv_xy
        self.idata.other.rsv_y1 = -self.idata.other.rsv_xy
        self.idata.other.rsv_y2 = self.idata.other.rsv_xy
        if self.idata.other.perm_frac:
            self.idata.other.frac_width = 10. # [m]
            self.idata.other.rsv_y1 = -self.idata.other.frac_width/2.
            self.idata.other.rsv_y2 = self.idata.other.frac_width/2.
            
        # rock properties for outside reservoir boundaries part of the mesh
        self.idata.rock.poro_non_rsv = 0.001
        #self.idata.rock.perm_non_rsv = 1e-9 # this matched thm and analytical solution
        self.idata.rock.perm_non_rsv = 0.01   # this matches proxy and thm
        self.idata.rock.E_non_rsv = self.idata.rock.E  # homogeneous geomech prop
        
        if self.idata.other.perm_frac:
            self.idata.rock.poro_non_rsv = 0.1
            self.idata.rock.perm_non_rsv = 1. # mD

        self.idata.rock.compressibility = get_rock_compressibility(
            kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu),
            biot=self.idata.rock.biot, poro0=self.idata.rock.porosity)
        print('bulk modulus = ', get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu))
        print('rock compressibility = ', self.idata.rock.compressibility)
        self.idata.rock.stiffness = get_isotropic_stiffness(self.idata.rock.E, self.idata.rock.nu)

        self.idata.rock.th_expn = 1e-5  # [1/K]
        self.idata.rock.th_expn_orig = self.idata.rock.th_expn  # save this for proxy
        self.idata.rock.th_expn *= get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu)  # Couchy book formula 4.19a, 4.21a
        self.idata.rock.th_expn *= 3. # Couchy book formula 4.22; from linear to volumetric
        
        self.idata.rock.thermal_conductivity = 260  # [kJ/m/day/K]
        self.idata.rock.heat_capacity = 2300  # [kJ/m3/K]

        self.idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update

        # Only for a single-phase physics
        self.idata.fluid.Mw = 18.015 # water molar weight, [g/mol]
        self.idata.fluid.compressibility = 4.4e-5  # [1/bar]
        self.idata.fluid.viscosity = 1.0  # [cP]
        self.idata.fluid.density = 1000. # [kg/m^3]
        
        # branch ilshat/fluid_heat_cond
        self.idata.fluid.thermal_conductivity = 0. # It is not used in the engine # [kJ/m/day/K] 
        #self.idata.fluid.heat_capacity = 2200. #[kJ/m3/K] - different unit than used for rock
        #self.idata.fluid.heat_capacity *= self.idata.fluid.Mw / self.idata.fluid.density  # convert from [kJ/m3/K] to [kJ/kmol/K]
        # water: 4170 [kJ/m3/K] or 75.37 [kJ/kmol/K]
        self.idata.fluid.heat_capacity = 75. #[kJ/kmol/K]

        # initial conditions (p, T gradients)
        
        # non-zero initial temperature doesn't work properly (doesn't converge, check t_ref implementation)
        self.idata.initial.reference_depth_for_temperature = 0.  # [m]
        self.idata.initial.temperature_gradient = 0.#0.03  # [K/m]
        self.idata.initial.temperature_at_ref_depth = 0.#273.15 + 10  # [K]
        
        # next 2 params don't affect the initial pressure since will be computed by equilibrium using fluid density
        # need to set well pressure controls as it is defined before the equilibrium state is evaluated
        self.idata.initial.pressure_gradient = 0.1  # [bar/m] # this is used only in reservoir.get_reservoir_initial_pressure() => reservoir.p_init 
        #self.idata.initial.reference_depth_for_pressure = 0.  # [m]
        self.idata.initial.pressure_at_ref_depth = 1.  # [bars]
    
        if self.physics_type == 'dead_oil' or self.physics_type == 'dead_oil_thermal':
            self.idata.initial.initial_composition = [0.67]

        # vertical well locations
        shift = 0. # if a single well - place to the center
        if self.wells_type == 'doublet':
            shift = 500. # half well ditance [m] 
        eps_perf = 1 # [m]
        perf_depth_start = self.idata.other.rsv_top + eps_perf
        perf_depth_end =  self.idata.other.rsv_bottom - eps_perf

        # as the perf is single, put it to the middle depth of the rsv
        perf_depth_start = (self.idata.other.rsv_top + self.idata.other.rsv_bottom)*0.5
        
        # 50 - to put into the cell center as (0,0) is a boundary between two cells
        self.idata.other.prod_well_coords = [50. - shift, 50., perf_depth_start, perf_depth_end] # X, Y, Z1, Z2
        self.idata.other.inj_well_coords = [50. + shift, 50., perf_depth_start, perf_depth_end] # X, Y, Z1, Z2
        self.well_init_depth = perf_depth_start

        # well controls
        self.idata.other.delta_temp_inj = 40 # [K] - delta for temperature control
        # RATE control
        self.idata.other.delta_p = None
        self.idata.other.wctrl_type = well_control_iface.MASS_RATE # mass or molar rate can be choosen here
        self.idata.other.well_rate = 2000. # [m^3/day] 
        self.idata.other.well_rate *= self.idata.fluid.density # [kg/day] unit depends on the type at the previous line

        self.idata.mesh.bnd_tags = {}
        tags = self.idata.mesh.bnd_tags  # short name
        tags['BND_X-'] = 991
        tags['BND_X+'] = 992
        tags['BND_Y-'] = 993
        tags['BND_Y+'] = 994
        tags['BND_Z-'] = 995
        tags['BND_Z+'] = 996
        mat_tag = 99991
        self.idata.mesh.matrix_tags = [mat_tag]
        # merge dicts (for mesh generation)
        self.idata.mesh.tags = self.idata.mesh.bnd_tags.copy()
        self.idata.mesh.tags['MATRIX_1'] = mat_tag

        if nx == 6: # for debugging, -4..4 km XY
            Xc = np.array([-4000, -2000, -1000, 0, 1000, 2000, 4000])
        elif nx == 16: # -4..4 km XY, dx = 100 m in the reservoir, outside 500-2000 m
            Xc = np.array([-4000, -2000, -1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000, 2000, 4000])
        elif nx == 42: # 
            Xc = np.array([-8000,-6000,-5000,-4000,-3000,-2500,-2000,-1600,-1400] + 
                          np.arange(-1200, 1200+1, 100).tolist() + 
                          [1400,1600,2000,2500,3000,4000,5000,6000,8000])
        elif nx == 34: # -15..15 km XY, dx = 100 m in the reservoir, outside 100-7000 m
            Xc = np.array([-15000,-8000,-4000,-2400,-1600,-1200,-1100,-1000] + np.arange(-900, 1000, 100).tolist() + [1000, 1100,1200, 1600, 2400, 4000,8000,15000])
        #elif nx == 41: # 41x41
        #    pass
        #    #rsv = np.arange(-900, 1000, 200)
        #    #side = np.arange(1000, 6500, 1000)
        #    #self.Xc = np.hstack([-side, rsv, side])
        else:
            print('not found an option to mesh with nx = ', nx)
            exit(1)

        rsv_top = self.idata.other.rsv_top
        rsv_bottom = self.idata.other.rsv_bottom
        if nz == 5: # for debugging
            Zc = np.array([0, 1000, 2000, 2100, 2200, 3000])
        elif nz == 15:  # dz = 100-1000 m for over and underburden and 20m for the reservoir
            Zc = np.array([0, 1000, 1500, 2000, 2100, 2120, 2140, 2160, 2180, 2200, 2300, 2500, 3000, 4000, 5000, 6000])
        elif nz == 29:  # dz = 200 m for over and underburden and 20m for the reservoir
            Zc = np.hstack([np.arange(0, rsv_top, 200), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 200)])
        elif nz == 37:  # dz = 200 m for over and underburden and 20m for the reservoir
            Zc = np.hstack([np.arange(0, rsv_top, 150), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 150)])
        elif nz == 53:  # dz = 100 m for over and underburden and 20m for the reservoir
            Zc = np.hstack([np.arange(0, rsv_top, 100), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 100)])
        elif nz == 57:  # refine a bit upper and lower (50m) reservoir as well, dz = 100 m for over and underburden and 25m for the reservoir
            Zc = np.hstack([np.arange(0, rsv_top - 100 + 1, 100),
                                 np.arange(rsv_top - 50, rsv_bottom + 50 + 1, 25),
                                 rsv_bottom + 100,
                                 np.arange(rsv_bottom + 200, 5000 + 1, 100)])
        elif nz == 66:  # refine a bit upper and lower (50m) reservoir as well, dz = 100 m for over and underburden and 25m for the reservoir
            Zc = np.hstack([np.arange(0, rsv_top - 100 + 1, 100),
                                 np.arange(rsv_top - 50, rsv_bottom + 50 + 1, 25),
                                 rsv_bottom + 100,
                                 np.arange(rsv_bottom + 200, 5000 + 1, 100)])
        elif nz == 90:  # refine a bit upper and lower (50m) reservoir as well, dz = 100 m for over and underburden and 25m for the reservoir
            Zc = np.hstack([np.arange(0, rsv_top - 500 + 1, 100),
                                 np.arange(rsv_top - 450, rsv_bottom + 450 + 1, 25),
                                 rsv_bottom + 500,
                                 np.arange(rsv_bottom + 600, 5000 + 1, 100)])
        else:
            print('not found an option to mesh with nz = ', nz)
            exit(1)
            
        # no frac
        Yc = Xc.copy()
        if self.idata.other.perm_frac:
            Yc = np.array([-15000,-8000,-4000,-2400,-1600,-1200,-1100,-1000] + np.arange(-900, 0, 100).tolist() + [-self.idata.other.frac_width/2., self.idata.other.frac_width/2.] + np.arange(100, 1000, 100).tolist() + [1000, 1100,1200, 1600, 2400, 4000,8000,15000])
        
        self.idata.other.Xc = Xc
        self.idata.other.Yc = Yc
        self.idata.other.Zc = Zc

        self.idata.obl.n_points = 400
        self.idata.obl.zero = 1e-9
        self.idata.obl.min_p = 0.0
        self.idata.obl.max_p = 1000.
        self.idata.obl.min_t = -50.#273.15
        self.idata.obl.max_t = 50.#273.15 + 300
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
        self.idata.obl.epsilon_z = 1e-10
        
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

            property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=self.idata.rock.heat_capacity, tref=t_ref))])
            property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0, tref=t_ref)  #TODO use hcap from idata? see https://gitlab.com/open-darts/open-darts/-/issues/19
            property_container.conductivity_ev = dict([('wat', ConstFunc(1.0))])
        elif self.physics_type == 'dead_oil' or self.physics_type == 'dead_oil_thermal':
            components = ['w', 'o']
            phases = ['wat', 'oil']
            self.cell_property = ['pressure'] + ['water']

            property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=self.idata.obl.epsilon_z)

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
                                      epsilon_z=self.idata.obl.epsilon_z,
                                      min_t=self.idata.obl.min_t, max_t=self.idata.obl.max_t,
                                      discretizer=self.discretizer_name)
        self.physics.add_property_region(property_container)

        self.physics.init_physics(discr_type=self.discretizer_name, platform='cpu')

        return

    def set_wells(self):
        #return
        centroids_3d = np.array([np.array([c.values[0], c.values[1], c.values[2]]) for
                              c in self.reservoir.discr_mesh.centroids])[:self.reservoir.n_matrix]

        if self.wells_type == 'none':
            well_names = []
            well_coords = []
        elif self.wells_type == 'prod':  # one well (prod)
            well_names = ['PRD1']
            well_coords = np.array([self.idata.other.prod_well_coords])
        elif self.wells_type == 'inj': # one well (inj)
            well_names = ['INJ1']
            well_coords = np.array([self.idata.other.inj_well_coords])
        elif self.wells_type == 'doublet':# two wells (doublet)
            well_names = ['PRD1', 'INJ1']
            well_coords = np.array([self.idata.other.prod_well_coords, self.idata.other.inj_well_coords])

        print('well_coords:', well_coords, 'well depth=', self.well_init_depth)
        print('centroids_mean depth:', centroids_3d[:, 2].mean())

        self.well_cell_ids = []

        nodes = np.array(self.reservoir.discr_mesh.nodes)
        elems = np.array(self.reservoir.discr_mesh.elems)

        step_z_perf = 1 # [m] should be smaller that cell dz
        
        for i, coord in enumerate(well_coords): # process each well
            # find mesh cells which 
            z1, z2 = coord[2], coord[3]
            z_points = np.arange(z1, z2, step_z_perf)
            ids = set()
            for z in z_points: # find a cell with the closest center
                cell = ((centroids_3d[:, 0] - coord[0]) ** 2 + (centroids_3d[:, 1] - coord[1]) ** 2 + (centroids_3d[:, 2] - z) ** 2).argmin()
                ids.add(int(cell))
            ids_1 = list(ids)
            
            # sort perforations by depth
            perf_depths = centroids_3d[ids_1, 2]
            perf_sorted_indices = np.argsort(perf_depths)
            ids_1 = np.array(ids_1)[perf_sorted_indices]
            
            self.well_cell_ids.append(ids_1)
            # adding a well
            self.reservoir.add_well(well_names[i], depth=self.well_init_depth)
            # adding perforations
            for cell_id in ids_1:
                cell = elems[cell_id]
                pt_ids = self.reservoir.discr_mesh.elem_nodes[cell.pts_offset:cell.pts_offset + cell.n_pts]
                pts = np.array([nodes[id].values for id in pt_ids])
                # Calculate well_index (very primitive way....):
                rw = 0.0762 # m
                # onstruct a right hexahedron around cell nodes
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
                print('well perf added to the cell', cell_id, 'with a center=', centroids_3d[cell_id], 'for the requested point=', centroids_3d[cell_id,:])
                break  #TODO add only one perforation for now, need to fix the issue with the crossflow 

    def set_boundary_conditions(self): # for initial mechanical equilibrium initialization, wells are switched off
        for i, w in enumerate(self.reservoir.wells):
                self.physics.set_well_controls(w.control,
                                               control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=False, target=0., phase_name='wat')


    def set_boundary_conditions_after_initialization(self): # set well controls
        #return
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:

        for i, w in enumerate(self.reservoir.wells):
            p_cell = self.initial_pressure[self.well_cell_ids[i][0]] # from the first perforation of the current well
            t_cell = self.initial_temperature[self.well_cell_ids[i][0]]

            delta_temp_inj = self.idata.other.delta_temp_inj
            delta_p = self.idata.other.delta_p
            well_rate = self.idata.other.well_rate
            wctrl_type = self.idata.other.wctrl_type
            
            if 'PRD' in w.name:
                target = p_cell - delta_p if wctrl_type == well_control_iface.BHP else well_rate
                print('prod well', w.name, 'control', wctrl_type, 'target', fmt(target))
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=wctrl_type,
                                               is_inj=False, 
                                               target=target)
            elif 'INJ' in w.name:
                inj = []
                inj_temp = None
                if self.physics_type == 'single_phase_thermal':
                    inj_temp = t_cell - delta_temp_inj
                elif self.physics_type == 'dead_oil':
                    inj = [1.0 - self.idata.obl.zero]
                elif self.physics_type == 'dead_oil_thermal':
                    inj = [1.0 - self.idata.obl.zero]
                    inj_temp = t_cell - delta_temp_inj
                target = p_cell + delta_p if wctrl_type == well_control_iface.BHP else well_rate
                print('inj well', w.name, 'control', wctrl_type, 'target ' + fmt(target), 'inj_temp = ' + fmt(inj_temp))
                self.physics.set_well_controls(wctrl=w.control, 
                                               control_type=wctrl_type,
                                               is_inj=True, 
                                               target=target, 
                                               inj_composition=inj,
                                               inj_temp=inj_temp)
        return 0

    def set_initial_conditions(self):      

        if True: # compute fluid equilibrium from given p,T at the surface
            # pressure gradient might vary as the density depends on the temperature
            boundary_state = {}
            if self.thermal:
                boundary_state['temperature'] = self.idata.initial.temperature_at_ref_depth 
            boundary_state['pressure'] = self.idata.initial.pressure_at_ref_depth
            init = Initialize(physics=self.physics, algorithm='multilinear', mode='adaptive',
                              is_barycentric=False)

            nb = 100
            nc = self.physics.nc
            z = np.zeros((nb, nc))
            z[:, 0] = 1 - self.idata.obl.zero  # liquid is below GOC
            primary_specs = {var: z[:, i] for i, var in enumerate(self.physics.components[:-1])}
            # run initialization
            min_depth = self.reservoir.depths.min()
            max_depth = self.reservoir.depths.max()
            X = init.solve_up_and_downwards(depth_bottom=max_depth, depth_top=min_depth, depth_known=min_depth,
                           nb=nb, primary_specs=primary_specs, boundary_state=boundary_state,
                           dTdh=self.idata.initial.temperature_gradient).reshape((nb, self.physics.n_vars))
            input_distribution = {var: X[:, i] for i, var in enumerate(self.physics.vars)}
            set_initial_conditions_from_depth_table(self=self.physics, mesh=self.reservoir.mesh, input_depth=init.depths,
                                                                 input_distribution=input_distribution,
                                                                 input_displacement=self.reservoir.u_init,
                                                            depths=self.reservoir.depths[:self.reservoir.mesh.n_blocks])
        else:
            #self.reservoir.p_init[:] = 500  # uniform init pressure
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
        s = np.asarray(self.reservoir.mesh.initial_state)
        if self.reservoir.thermoporoelasticity:
            self.initial_pressure = s[0::2][:self.reservoir.n_matrix]
            self.initial_temperature = s[1::2][:self.reservoir.n_matrix]
        else:
            self.initial_pressure = s[:self.reservoir.n_matrix]
            self.initial_temperature = np.zeros_like(s)[:self.reservoir.n_matrix]
        print('Initial pressure: min/mean/max:', fmt(self.initial_pressure.min()), fmt(self.initial_pressure.mean()), fmt(self.initial_pressure.max()))
        print('Initial temperature: min/mean/max:', fmt(self.initial_temperature.min()), fmt(self.initial_temperature.mean()), fmt(self.initial_temperature.max()))

        centroids_3d = np.array([np.array([c.values[0], c.values[1], c.values[2]]) for
                              c in self.reservoir.discr_mesh.centroids])[:self.reservoir.n_matrix]
        eps = 1
        from functools import reduce
        rsv = reduce(np.logical_and, [self.reservoir.rsv_top - eps < centroids_3d[:,2], centroids_3d[:,2] < self.reservoir.rsv_bottom + eps])
        print('Initial pressure rsv: min/mean/max:', fmt(self.initial_pressure[rsv].min()), fmt(self.initial_pressure[rsv].mean()), fmt(self.initial_pressure[rsv].max()))
        print('Initial temperature rsv: min/mean/max:', fmt(self.initial_temperature[rsv].min()), fmt(self.initial_temperature[rsv].mean()), fmt(self.initial_temperature[rsv].max()))
    
        return 0


def set_initial_conditions_from_depth_table(self, mesh, input_distribution: dict, input_displacement: dict,
                                            input_depth, depths):
    """
    Function to set initial conditions from given distribution of properties over depth.

    :param mesh: conn_mesh object
    :param input_distribution: Initial distributions of unknowns over depth, must have keys equal to self.vars
                               and each entry is scalar or array of length equal to depths
    :param input_depth: Array of depths over which depth table has been specified
    """
    # Assertions of consistent depth table specification
    assert np.all([variable in input_distribution.keys() for variable in self.vars[1:self.nc]]), \
        "Initial state for must be specified for all primary variables"
    assert not self.thermal or ('temperature' in input_distribution.keys() or
                                'enthalpy' in input_distribution.keys()), \
        "Temperature or enthalpy must be specified for thermal models"
    input_depth = input_depth if hasattr(input_depth, "__len__") else np.array([input_depth])
    for key, input_values in input_distribution.items():
        input_values = input_values if hasattr(input_values, "__len__") else np.ones(len(input_depth)) * input_values
        assert len(input_values) == len(input_depth)

    # Get depths and primary variable arrays from mesh object
    #depths = np.asarray(mesh.depth)[:mesh.n_blocks]

    # adjust the size of initial_state array in c++
    mesh.initial_state.resize(mesh.n_blocks * self.n_vars)

    # Loop over variables to fill initial_state vector in c++
    for ith_var, variable in enumerate(self.vars):
        if variable == "enthalpy" and "enthalpy" not in input_distribution.keys():
            # If temperature has been provided, interpolate pressure and temperature to compute enthalpies
            p_itor = interp1d(input_depth, input_distribution['pressure'], kind='linear', fill_value='extrapolate')
            pressure = p_itor(depths)

            t_itor = interp1d(input_depth, input_distribution['temperature'], kind='linear', fill_value='extrapolate')
            temperature = t_itor(depths)

            values = np.empty(mesh.n_blocks)
            for j in range(mesh.n_blocks):
                state = np.array([pressure[j], temperature[j]])
                values[j] = self.property_containers[0].compute_total_enthalpy(state, temperature[j])
        else:
            # Else, interpolate primary variable
            itor = interp1d(input_depth, input_distribution[variable], kind='linear', fill_value='extrapolate')
            values = itor(depths)

        np.asarray(mesh.initial_state)[ith_var::self.n_vars] = values

    # set initial displacements
    for i in range(self.n_dim):
        np.asarray(mesh.displacement)[i::self.n_dim] = input_displacement[i]


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
