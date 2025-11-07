from darts.engines import well_control_iface
from darts.input.input_data import InputData
from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH, GeothermalIAPWSFluidProps, GeothermalPHFluidProps

def set_input_data_geothermal():
    # init_type = 'uniform'
    init_type = 'gradient'
    idata = InputData(type_hydr='thermal', type_mech='none', init_type=init_type)

    idata.other.iapws_physics = True
    if idata.other.iapws_physics:
        idata.fluid = GeothermalIAPWSFluidProps()
    else:
        idata.fluid = GeothermalPHFluidProps()

    # example - how to change the properties
    # idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)

    # from darts.physics.properties.basic import ConstFunc
    # idata.fluid.conduction_ev['water'] = ConstFunc(172.8)

    if init_type == 'uniform':  # uniform initial conditions
        idata.initial.initial_pressure = 200.  # bars
        idata.initial.initial_temperature = 350.  # K
    elif init_type == 'gradient':  # gradient by depth
        idata.initial.reference_depth_for_pressure = 0  # [m]
        idata.initial.pressure_gradient = 100  # [bar/km]
        idata.initial.pressure_at_ref_depth = 1  # [bars]

        idata.initial.reference_depth_for_temperature = 0  # [m]
        idata.initial.temperature_gradient = 30  # [K/km]
        idata.initial.temperature_at_ref_depth = 273.15 + 20  # [K]

    idata.obl.n_points = 100
    idata.obl.min_p = 50.
    idata.obl.max_p = 400.
    idata.obl.min_e = 1000.  # kJ/kmol, will be overwritten in PHFlash physics
    idata.obl.max_e = 25000.  # kJ/kmol, will be overwritten in PHFlash physics

    return idata

def set_input_data_well_controls_geothermal(idata: InputData, case: str):
    # well controls
    wdata = idata.well_data
    wells = wdata.wells  # short name

    if 'bhp' in case:
        for w in wells:
            if idata.well_is_inj(w):
                wdata.add_inj_bhp_control(name=w, bhp=250, temperature=300)  # m3/day | bars | K
            else:  # prod
                wdata.add_prd_bhp_control(name=w, bhp=100)  # m3/day | bars
    elif '_rate' in case:
        for w in wells:
            if idata.well_is_inj(w):
                wdata.add_inj_rate_control(name=w, rate=5500, rate_type=well_control_iface.VOLUMETRIC_RATE,
                                           bhp_constraint=300, temperature=300)  # m3/day | bars | K
            else:  # prod
                wdata.add_prd_rate_control(name=w, rate=5500, rate_type=well_control_iface.VOLUMETRIC_RATE,
                                           bhp_constraint=70)  # m3/day | bars
    elif 'periodic' in case:
        wname = list(wdata.wells.keys())[0]  # single well
        y2d = 365.25
        for i in range(0, len(idata.sim.time_steps), 4):
            # iterate [inj - stop - prod - stop]
            wdata.add_inj_rate_control(time=(i + 0) * y2d, name=wname, rate=5500,
                                       rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=300,
                                       temperature=300)
            wdata.add_prd_rate_control(time=(i + 1) * y2d, name=wname, rate=0,
                                       rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=5)
            wdata.add_prd_rate_control(time=(i + 2) * y2d, name=wname, rate=5500,
                                       rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=5)
            wdata.add_prd_rate_control(time=(i + 3) * y2d, name=wname, rate=0,
                                       rate_type=well_control_iface.VOLUMETRIC_RATE, bhp_constraint=5)
    else:
        raise ValueError('case not recognized for well controls', case)
