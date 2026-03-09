from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import value_vector, sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from properties import STARSFoamRelPerm, WaterPhaseRelPerm

# from convertInjection import Conversion

class Model(CICDModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=1e-5, mult_ts=2, max_ts=1e-3, runtime=300, tol_newton=1e-3, tol_linear=1e-6)

        # find saturation corresponding with composition
        # INITIAL CONDITION
        self.sgr = 0.293
        z_range = np.linspace(self.zero, 1 - self.zero, 10000)
        for z in z_range:
            state = [40, z]
            # sat = self.physics.property_containers[0].compute_saturation_full(state)
            sat = self.physics.property_containers[0].evaluate(state) 
            if self.physics.property_containers[0].sat[self.physics.phases.index("wat")] >= 1. - self.sgr: # sgr = 0.293 -> S_w^0 = 1. - sgr
                self.ini = value_vector([z])
                break
        print("S_w^0 = ", self.ini)

        # INJECTION CONDITION
        # self.conv = Conversion()
        self.fg = 0.1
        self.vol_rate = 0.001436 # [m3/day]    
        # self.set_fgInj(fg=fg)  
        # print("S_w^inj = ", self.inj_stream)   

        self.timer.node["initialization"].stop()

    def set_fg_Inj(self, fg):
        self.fg = fg
        self.set_well_controls()

    # def set_fgInj(self, fg):
    #     self.fw = 1. - self.fg
    #     self.inj_gas_rate = self.conv.gas_inj_rate(self.fg*self.vol_rate, 40) # convert m3/day to kmol/day 
    #     self.inj_wat_rate = self.conv.wat_inj_rate(self.fw*self.vol_rate) # convert m3/day to kmol/day

    #     # Find Sw_inj from fg
    #     Sw_inj = 0.
    #     Sw_range = np.linspace(self.zero, 1 - self.zero, 10000)
    #     for Sw in Sw_range:
    #         krg_foam = self.physics.property_containers[0].rel_perm_ev['gas'].evaluate(Sw)
    #         krw = self.physics.property_containers[0].rel_perm_ev['wat'].evaluate(Sw)
    #         mu_g = self.physics.property_containers[0].viscosity_ev['gas'].evaluate()
    #         mu_w = self.physics.property_containers[0].viscosity_ev['wat'].evaluate()
            
    #         lambda_g = krg_foam/mu_g
    #         lambda_w = krw/mu_w
    #         lambda_t = lambda_w + lambda_g
    #         fg = lambda_g / lambda_t

    #         if abs(fg - self.fg) <= 1e-3:
    #             Sw_inj = Sw
    #             break
    #     # Find composition from Sw_inj
    #     z_range = np.linspace(self.zero, 1 - self.zero, 10000)
    #     for z in z_range:
    #         state = [40, z]
    #         sat = self.physics.property_containers[0].evaluate(state) 
    #         if self.physics.property_containers[0].sat[self.physics.phases.index("wat")] >= Sw_inj:
    #             self.inj_stream = value_vector([z])
    #             break

    def set_reservoir(self):
        nx = 500
        L = 0.15
        permx = np.ones(nx)*273.5775
        permx[0] = 1e10
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=L/nx, dy=0.033853868, dz=0.033853868,
                                         permx=permx, permy=300, permz=300, poro=0.25, hcap=0, rcond=0, depth=100)
        # self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=0.0015, dy=0.033853868, dz=0.033853868,      # TODO: is there a min value for dy, dz?
                                        #  permx=273.5775, permy=300, permz=300, poro=0.25, hcap=0, rcond=0, depth=100)                                
        
        
        # self.reservoir.init_reservoir()
        # volume = np.array(self.reservoir.mesh.volume, copy=False)
        # volume[-1] = 1e8
        return

    def set_wells(self):
        self.reservoir.add_well("I1") 
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1), well_index=1e5)

        # TODO: Isn't possible to define 2 wells at same cell_index? when I set cell_index=(2, 1, 1) it works!
        self.reservoir.add_well("I2") # wat well
        self.reservoir.add_perforation("I2", res_cell_idx=(2, 1, 1), well_index=1e5)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1), well_index=1e5)

    def set_physics(self):
        """Physical properties"""
        self.zero = 1e-13
        components = ["w", "g"]
        phases = ["wat", "gas"]

        property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=self.zero/10)

        property_container.density_ev = dict([('wat', DensityBasic(compr=0.0, dens0=55.5)),
                                              ('gas', DensityBasic(compr=0.0, dens0=22.3))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.5)),
                                                ('gas', ConstFunc(0.02112))])
        pfoam = np.array([269.34, 0.4376, 787.4579])
        property_container.rel_perm_ev = dict([('wat', WaterPhaseRelPerm("wat", swc=0.4, sgr=0.293, kre=0.302, n=2.98)),
                                            #    ('gas', PhaseRelPerm("gas", swc=0.4, sgr=0.293, kre=0.302, n=2.98))])
                                               ('gas', STARSFoamRelPerm("gas", pfoam, swc=0.4, sgr=0.293, kre=0.04, n=0.96))])
        
        # create physics
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=400, min_p=0, max_p=1000, epsilon_z=self.zero/10, min_z=self.zero, max_z=1 - self.zero)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 40., # [Bar] ???
                              self.physics.vars[1]: self.ini[0],
                             }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        # print(self.reservoir.wells[2].name)
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:   # GAS INJECTION WELL
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=self.fg*self.vol_rate, phase_name='gas', inj_composition=value_vector([self.zero])) 
            elif i == 1: # WATER INJECTION WELL
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=(1. - self.fg)*self.vol_rate, phase_name='wat', inj_composition=value_vector([1.-self.zero]))    
            else:      # PRODUCTION WELL
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=40.)

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, eps_z=eps_z, temperature=1.)

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
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

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

        return

    # def evaluate_at_cond(self, pressure, zc):
    #     self.sat[:] = 0

    #     ph = [0, 1]
    #     for j in ph:
    #         self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

    #     self.dens_m = [1025, 0.77]  # to match DO based on PVT

    #     self.nu = zc
    #     self.compute_saturation(ph)

    #     return self.sat, self.dens_m
