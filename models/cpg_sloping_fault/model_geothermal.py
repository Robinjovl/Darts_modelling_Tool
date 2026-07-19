import numpy as np
import pandas as pd

from darts.engines import value_vector, sim_params, well_control_iface

from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from dartsflash.mixtures import DARTSFlash, CompData, EoS, IAPWS
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm, RockCompactionEvaluator
from darts.physics.properties.viscosity import MaoDuan2009

from darts.input.input_data import InputData
from set_case import set_input_data
from model_cpg import Model_CPG, fmt


class ModelGeothermal(Model_CPG):
    def __init__(self, iapws_physics: bool = True):
        self.iapws_physics = iapws_physics
        super().__init__()
        # The OBL grid is unbounded (no axes_max) in this branch, so a producer
        # well-control switch can push the well-block enthalpy outside the IAPWS-valid
        # range in a single Newton step -> singular CPR system -> NaN runaway -> crash.
        # Enable the global Newton chop (caps the per-step relative change of all
        # variables) to damp that transient.
        self.set_solver()
        self.nonlinear_solver.spec.chop.mode = 'global'
        self.nonlinear_solver.spec.chop.factor = 0.2

    def set_physics(self):
        # Single component, two phase. Uses the compositional engine in PT-flash mode
        # with IAPWS EoS (drop-in replacement for the legacy Geothermal physics).
        # State vector layout is [P, T] (n_vars=2).
        self.set_iapws_physics(
            p_step=self.idata.obl.p_step,
            p_origin=self.idata.obl.p_origin,
            t_step=self.idata.obl.t_step,
            t_origin=self.idata.obl.t_origin,
        )

    def set_iapws_physics(self, p_step, p_origin, t_step, t_origin, cache=False):
        """Drop-in replacement for legacy Geothermal(...) using compositional + IAPWS PT-flash.
        Single-component water; phases are vapor ('V') and liquid ('L').
        State spec is PT so engine.X layout is [P, T, ...] and the OBL grid is sampled on (P, T).
        """
        components = ["H2O"]
        phases = ['V', 'L']
        zero = 1e-12
        comp_data = CompData(components=components, setprops=True)

        pc = PropertyContainer(phases_name=phases, components_name=components,
                               Mw=comp_data.Mw, eps_z=zero)

        pc.rock_compr_ev = RockCompactionEvaluator(pref=self.idata.rock.compressibility_ref_p,
                                                   compres=self.idata.rock.compressibility)

        flash_ev = IAPWS(iapws_ideal=True, ice_phase=False)
        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash)
        pc.flash_ev = flash_ev

        pc.density_ev = {
            'V': EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX),
            'L': EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MIN),
        }
        pc.viscosity_ev = {
            'V': ConstFunc(0.01),                # cP, steam
            'L': MaoDuan2009(components),        # cP, liquid water (pressure/temperature-dependent)
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
            'L': ConstFunc(172.8),       # kJ/m/day/K, matches geothermal default
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
        if self.idata.initial.type == 'gradient':
            # Specify reference depth, values and gradients to construct depth table in super().set_initial_conditions()
            input_depth = [0., np.amax(self.reservoir.mesh.depth)]
            input_distribution = {'pressure': [1., 1. + input_depth[1] * self.idata.initial.pressure_gradient/1000],
                                  'temperature': [293.15, 293.15 + input_depth[1] * self.idata.initial.temperature_gradient/1000]
                                  }
            g2l = np.asarray(self.reservoir.discr_mesh.global_to_local)[:self.reservoir.mesh.n_res_blocks]
            return self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                                        input_distribution=input_distribution,
                                                                        input_depth=input_depth)
        elif self.idata.initial.type == 'uniform':
            input_distribution = {'pressure': self.idata.initial.initial_pressure,
                                  'temperature': self.idata.initial.initial_temperature}
            return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                                  input_distribution=input_distribution)



    def get_arrays(self):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        a = self.reservoir.input_arrays  # include initial arrays and the grid

        nv = self.physics.n_vars
        n_ops = self.output.n_ops
        nb = self.reservoir.mesh.n_res_blocks
        Xn = np.array(self.physics.engine.X, copy=False)
        state = value_vector(Xn.T.flatten())

        # Interpolate temperature with property interpolator
        values = value_vector(np.zeros(n_ops * nb))
        values_numpy = np.array(values, copy=False)
        dvalues = value_vector(np.zeros(n_ops * nb * nv))
        i = 0
        for region, prop_itor in self.physics.property_itor.items():
            prop_itor.evaluate_with_derivatives(state, self.physics.engine.region_cell_idx[i], values, dvalues)
            i += 1

        # Get P from state vector and T from interpolated properties
        P = np.array(state[0:nb*nv:nv])
        T = values_numpy[0:nb*n_ops:n_ops]
        T -= 273.15  # K to degrees

        a.update({'PRESSURE': P, 'TEMPERATURE': T})

        print('P range [bars]:', fmt(P.min()), '-', fmt(P.max()), 'T range [degrees]:', fmt(T.min()), '-', fmt(T.max()))

        return a

    def print_well_rate(self):
        inj_well = prd_well = None
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):
                inj_well = w
            else:
                prd_well = w
        time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
        years = np.array(time_data['time'])[-1]/365.25

        rate_inj = rate_prd = temp_prd = temp_inj = 0.
        # compositional engine emits per-phase rate columns as "<name> : <phase> rate (m3/day)".
        # We use the liquid phase ('L') for water rate. Temperature column key is unchanged.
        if prd_well is not None:
            pr_col_name = time_data.filter(like=prd_well.name + ' : L rate').columns.to_list()
            pt_col_name = time_data.filter(like=prd_well.name + ' : temperature').columns.to_list()
            rate_prd = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
            temp_prd = np.array(time_data[pt_col_name])[-1][0]  # pick the last timestep value
        if inj_well is not None:
            ir_col_name = time_data.filter(like=inj_well.name + ' : L rate').columns.to_list()
            it_col_name = time_data.filter(like=inj_well.name + ' : temperature').columns.to_list()
            rate_inj  = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
            temp_inj = np.array(time_data[it_col_name])[-1][0]  # pick the last timestep value
        print(fmt(years), 'years:', 'RATE_prod =', fmt(rate_prd), 'RATE_inj =', fmt(rate_inj), 'TEMP_prod =', fmt(temp_prd), 'TEMP_inj =', fmt(temp_inj))

    def set_input_data(self, case=''):
        #init_type = 'uniform'
        init_type = 'gradient'
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type=init_type)

        self.idata.other.iapws_physics = True

        set_input_data(self.idata, case)

        # Fluid evaluators are now wired directly inside set_iapws_physics() via the
        # compositional PropertyContainer; no idata.fluid assignment is required.

        if init_type== 'uniform': # uniform initial conditions
            self.idata.initial.initial_pressure = 200.  # bars
            self.idata.initial.initial_temperature = 350.  # K
        elif init_type == 'gradient':         # gradient by depth
            self.idata.initial.reference_depth_for_pressure = 0  # [m]
            self.idata.initial.pressure_gradient = 100  # [bar/km]
            self.idata.initial.pressure_at_ref_depth = 1 # [bars]

            self.idata.initial.reference_depth_for_temperature = 0  # [m]
            self.idata.initial.temperature_gradient = 30  # [K/km]
            self.idata.initial.temperature_at_ref_depth = 273.15 + 20 # [K]

        # well controls
        wdata = self.idata.well_data
        wells = wdata.wells  # short name

        if 'wbhp' in case:
            for w in wells:
                if self.well_is_inj(w):
                    wdata.add_inj_bhp_control(name=w, bhp=250, temperature=300)  # m3/day | bars | K
                else: # prod
                    wdata.add_prd_bhp_control(name=w, bhp=100) # m3/day | bars
        elif 'wrate' in case:
            for w in wells:
                if self.well_is_inj(w):
                    wdata.add_inj_rate_control(name=w, rate=5500, rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=300, temperature=300, phase_name='L')  # m3/day | bars | K
                else: # prod
                    wdata.add_prd_rate_control(name=w, rate=5500, rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=70, phase_name='L') # m3/day | bars
        elif 'wperiodic' in case:
            wname = list(wdata.wells.keys())[0]  # single well
            y2d = 365.25
            for i in range(0, len(self.idata.sim.time_steps), 4):
                # iterate [inj - stop - prod - stop]
                wdata.add_inj_rate_control(time=(i+0)*y2d, name=wname, rate=5500, rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=300, temperature=300, phase_name='L')
                wdata.add_prd_rate_control(time=(i+1)*y2d, name=wname, rate=0,    rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=5, phase_name='L')
                wdata.add_prd_rate_control(time=(i+2)*y2d, name=wname, rate=5500, rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=5, phase_name='L')
                wdata.add_prd_rate_control(time=(i+3)*y2d, name=wname, rate=0,    rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=5, phase_name='L')
        else:
            assert False, 'Unknown wctrl_type' +  case

        # OBL grid for the compositional + IAPWS PT-flash physics (state_spec=PT).
        # Cell sizes reproduce the former 100-point grid over p in [50, 400] bar and
        # T in [250, 575] K; the adaptive interpolator extends past it on demand.
        self.idata.obl.p_step = 3.5   # bar
        self.idata.obl.p_origin = 50.0
        self.idata.obl.t_step = 3.25  # K
        self.idata.obl.t_origin = 250.0
