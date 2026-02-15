from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import DartsModel
from darts.engines import sim_params, well_control_iface
from darts.engines import (
    conn_mesh,
    index_vector,
    ms_well,
    ms_well_vector,
    timer_node,
    value_vector,
)
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, Garcia2001
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.physics.properties.viscosity import Fenghour1998, Islam2012  
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from darts.physics.super.initialize import Initialize


class Model(DartsModel):
    def build_dz (self, dz0:float, total_thickness: float, ratio: float = 2.0):
        layers = []
        s = 0
        dz = float(dz0)
        while dz + s < total_thickness:
            layers.append(dz)
            s += dz
            dz*= ratio
        layers.append(total_thickness - s) # add last layer
        return np.asarray(layers, dtype=float)
    def build_cell_center(self):
        if not hasattr(self, 'reservoir') or self.reservoir is None:
            raise RuntimeError("Reservoir not built yet.")
        n = self.reservoir.n
        x = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        z = np.asarray(self.reservoir.global_data["depth"], dtype = float).copy()
        # level 0 
        nx0= int(self.reservoir.nx)
        ny0= int(self.reservoir.ny)
        nz0= int(self.reservoir.nz)
        dx0 =  float(np.asarray(self.reservoir.global_data["dx"]).flat[0])
        dy0 = float(np.asarray(self.reservoir.global_data["dy"]).flat[0])
        for id in range(n):
            k = id // (nx0 * ny0) # 0 based
            j = (id % (nx0 * ny0)) // nx0
            i = id % nx0
            x[id] = (i + 0.5) * dx0
            y[id] = (j + 0.5) * dy0

        self.reservoir.cell_center_x = x
        self.reservoir.cell_center_y = y
        self.reservoir.cell_center_z = z
        return x,y,z
        
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()

        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=30, runtime=1000, 
                            tol_newton=1e-3, tol_linear=1e-3,
                            it_newton=10, it_linear=50)

        self.timer.node["initialization"].stop()
    
    def set_reservoir(self):

        nx0, ny0 = 80*3, 80*3 # global grid size
        dx0, dy0 = 100/3, 100/3
        nz_res = 40
        dz_res = 5

        over_thickness = 2000.0
        under_thickness = 2000.0
        dz_over = self.build_dz(dz0=dz_res, total_thickness=over_thickness)
        dz_over = dz_over[::-1]  # reverse for overburden

        dz_under = self.build_dz(dz0=dz_res,total_thickness=under_thickness)
        nz_over = len(dz_over)
        nz_under = len(dz_under)
        nz0 = nz_over + nz_res + nz_under
        dz0_layers = np.concatenate([dz_over, np.full(nz_res, dz_res, dtype=float), dz_under])
        permx0, permy0, permz0 = 50, 50, 50
        poro0 = 0.1
        poro_burden = 0.0001
        perm_burden = 1e-6
        self.nz_over = nz_over
        self.nz_res = nz_res

        # thermal properties
        rcond_res = 181.44 # KJ/m/day/k
        hcap_res = 2650 # kJ/m3/K assume reservoir density here
        rcond_over, rcond_under = 149.54, 149.54
        hcap_over, hcap_under = 2347.29, 2347.29

        k_index0 = np.arange(nx0 * ny0 * nz0, dtype=np.int32) // (nx0 * ny0)
        kx0_full = np.full(nx0 * ny0 * nz0, perm_burden, dtype=float)
        ky0_full = np.full(nx0 * ny0 * nz0, perm_burden, dtype=float)
        kz0_full = np.full(nx0 * ny0 * nz0, perm_burden, dtype=float)

        rcon0_full = np.empty(nx0 * ny0 * nz0, dtype=float)
        hcap0_full = np.empty(nx0 * ny0 * nz0, dtype=float)
        poro0_full = np.full(nx0 * ny0 * nz0, poro_burden, dtype=float)
  
        mask_over = k_index0 < nz_over
        mask_res  = (k_index0 >= nz_over) & (k_index0 < nz_over + nz_res)
        mask_under = k_index0 >= (nz_over + nz_res)
        kx0_full [mask_res] = permx0
        ky0_full [mask_res] = permy0
        kz0_full [mask_res] = permz0

        rcon0_full[mask_over] = rcond_over
        rcon0_full[mask_res] = rcond_res
        rcon0_full[mask_under] = rcond_under
        hcap0_full[mask_over] = hcap_over
        hcap0_full[mask_res] = hcap_res
        hcap0_full[mask_under] =hcap_under
        poro0_full[mask_res] = poro0

        self.reservoir = StructReservoir(self.timer, nx=nx0, ny=ny0, nz=nz0, dx=dx0, dy=dy0, dz=dz0_layers,
                                      permx=kx0_full, permy=ky0_full, permz=kz0_full, poro=poro0_full,depth= None, 
                                      start_z=0, rcond=rcon0_full, hcap=hcap0_full,)
        boundary_factor = 2000
        base_vol = float(dx0 * dy0 * dz_res)
        v_big = base_vol * boundary_factor

        self.reservoir.boundary_volumes = {
            "xy_minus": None,
            "xy_plus": None,
            "yz_minus": v_big,
            "yz_plus": v_big,
            "xz_minus": v_big,
            "xz_plus": v_big,
        }
        self.reservoir.discretize()
        self.build_cell_center()
        return

   



    def set_wells(self):
        self.reservoir.add_well("I1")
        for k in range(self.nz_over, self.nz_over + self.nz_res):         
            self.reservoir.add_perforation("I1", res_cell_idx=(60,120,k))

        self.reservoir.add_well("P1")
        for k in range(self.nz_over, self.nz_over + self.nz_res):         
            self.reservoir.add_perforation("P1", res_cell_idx=(180,120,k))
  

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        components = ['CO2', 'H2O']
        self.components = components
        comp_data = CompData(components, setprops=True)
        phases = ['CO2_rich', 'aqueous']
        # Mw = [44.01, 18.015]

        ceos = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                                AQEoS.solute: AQEoS.Ziabakhsh2012,
                                })
        flash_params = FlashParams(comp_data)
        # EoS-related parameters
        flash_params.add_eos("CEOS", ceos)
        flash_params.add_eos("aqueous", aq)
        flash_params.eos_order = ["aqueous", "CEOS"]

        # Flash-related parameters
        flash_params.split_tol = 1e-12

        property_container = PropertyContainer(phases_name=phases, components_name=components,
                                               Mw=comp_data.Mw, min_z=self.zero / 10,)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(components), [4, 1e-1], self.zero)
        # property_container.flash_ev = NegativeFlash(flash_params, ["aqueous", "CEOS"], [InitialGuess.Henry_AV])
        property_container.density_ev = dict([('CO2_rich', EoSDensity(ceos,comp_data.Mw)),
                                              ('aqueous', DensityBasic(dens0=1020,compr= 4.5e-5,p0=31))])
        property_container.viscosity_ev = dict([('CO2_rich', Fenghour1998()),
                                                ('aqueous', Islam2012(components))])
        property_container.rel_perm_ev = dict([('CO2_rich', PhaseRelPerm("gas")),
                                               ('aqueous', PhaseRelPerm("wat"))])
        property_container.enthalpy_ev = dict([('CO2_rich', EoSEnthalpy(ceos)),
                                                ('aqueous', EoSEnthalpy(aq))])
        property_container.conductivity_ev = dict([('CO2_rich', ConstFunc(10.)),
                                                   ('aqueous', ConstFunc(180.)), ])

      

        """ Activate physics """
        thermal = True
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=400, min_p=1, max_p=1000, min_z=self.zero/10, max_z=1-self.zero/10,
                                     min_t=273.15, max_t=373.15+200)


        property_container.output_props = {
            "satG": lambda: property_container.sat[0],
            "temp": lambda: property_container.temperature,
            "rhoG": lambda: property_container.dens[0],
            "rhoAq": lambda: property_container.dens[1],
            }

        self.physics.add_property_region(property_container) 

        return

    def set_initial_conditions(self):
        # input_distribution = {self.physics.vars[0]: 195, # pressure
        #                       self.physics.vars[1]: self.zero, # z_CO2
        #                       self.physics.vars[2]: 353.15 # temperature
        #                       }
        # return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, 
        #                                                       input_distribution=input_distribution)

        depths = np.asarray(self.reservoir.mesh.depth)
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        nb = int(self.reservoir.nz)
        depths = np.linspace(min_depth,max_depth,nb)

        init = Initialize(self.physics)
   
        primary_specs = {}
        for comp in self.physics.components[:-1]:
            primary_specs[comp] = np.ones(nb) * self.zero
        
        boundary_state = {"pressure" :195}  
        for comp in self.physics.components[:-1]:
            boundary_state[comp] = float(primary_specs[comp][0])
        boundary_state["temperature"] = 80 +273.15

        dTdh = 40/1000 #k/m
        # need to be fixed, currently reference T/P was set at the top of the overburden, which is wrong
        X = init.solve(depth_bottom=max_depth, depth_top= min_depth,depth_known=2000, nb=nb,
                       boundary_state=boundary_state,primary_specs=primary_specs,secondary_specs=None, dTdh=dTdh).reshape((nb, self.physics.n_vars))
        # input_distribution = {self.physics.vars[0]: 131, # pressure
        #                       self.physics.vars[1]: self.zero, # z_CO2
        #                       self.physics.vars[2]: 353.15 # temperature
        #                       }
        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                             input_depth= init.depths,
                                                            input_distribution={v:X[:,i] for i, v in enumerate(self.physics.vars)})
        return
    
        # self.reservoir.mesh.volume[0:3] = 1e20

    def set_well_controls(self):
        inj_composition = [1.0 - self.zero]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                #                                is_inj=True, target=5600000., inj_composition=inj_composition, inj_temp=296.15)
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=360., inj_composition=inj_composition, inj_temp=296.15)
            else:
                # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                #                                is_inj=False, target=1400000.)
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=102.)



   