from darts.input.input_data import InputData
from darts.engines import value_vector
from darts.physics.deadoil import DeadOil, DeadOil2PFluidProps
from darts.engines import well_control_iface

def set_input_data_deadoil():
    idata = InputData(type_hydr='isothermal', type_mech='none', init_type='uniform')

    # this sets default properties
    idata.fluid = DeadOil2PFluidProps() # if twophase else DeadOil3PFluidProps

    # example - how to change the properties
    # idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)

    idata.obl.n_points = 400
    idata.obl.zero = 1e-13
    idata.obl.min_p = 0.
    idata.obl.max_p = 1000.
    idata.obl.min_t = 10.
    idata.obl.max_t = 100.
    idata.obl.min_z = idata.obl.zero
    idata.obl.max_z = 1 - idata.obl.zero

    return idata

def set_input_data_well_controls_deadoil(idata: InputData, case: str):
    # well controls
    wdata = idata.well_data
    wells = wdata.wells  # short name
    # set default injection composition
    inj_comp = value_vector([idata.obl.zero * 100])  # injection composition - water

    if 'bhp' in case:
        for w in wells:
            if idata.well_is_inj(w):
                wdata.add_inj_bhp_control(name=w, bhp=250, phase_name='water', inj_composition=inj_comp, temperature=300)  # kmol/day | bars | K
            else:  # prod
                wdata.add_prd_bhp_control(name=w, bhp=100)  # kmol/day | bars
    elif 'rate' in case:
        for w in wells:
            if idata.well_is_inj(w): # inject water
                wdata.add_inj_rate_control(name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE, phase_name='water',
                                           inj_composition=inj_comp, bhp_constraint=250)  # kmol/day | bars | K
            else:  # prod
                wdata.add_prd_rate_control(name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE, phase_name='oil', bhp_constraint=100)  # kmol/day | bars
    else:
        raise ValueError('case not recognized for well controls', case)