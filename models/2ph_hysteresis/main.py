import numpy as np
import matplotlib.pyplot as plt
from darts.engines import value_vector, redirect_darts_output,sim_params

from model import Model, PhaseRelPermTable, Corey,layer_props
import os

property_regions  = [0, 1]
layers_to_regions = {"1": 0, "2": 1}
m = Model()
p_prodlist = [250.]
T_inj = 338.
fig, bx = plt.subplots()
continuous = True
for p_prod in p_prodlist:
    # output_folder = f"Production pressure = {p_prod}bar"
    n_points = 1000
    output_folder = f"1D_continuous_hys{n_points}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create folder if it doesn't exist
    log_filename = os.path.join(output_folder, "binary.log")
    redirect_darts_output(log_filename)
    if 1 :
        m.well_centers = {
            "I1": [0.0, 0.0, 0.5]}
    elif 0:
        m.well_centers = {
            "I1": [[0.0, 0.0, z] for z in np.arange(0.5, 10.1, 1.0)],
            # "P1": [[0.0, 0.0, z] for z in np.arange(0.5, 10.1, 1.0)]
        }
    m.rate_rhs = True
    theraml = False
    m.hys = True
    m.prod = True
    m.set_reservoir(100,layer_props = layer_props, layers_to_regions = layers_to_regions,logscale=False, rad_grid=False)  # set parameters of reservoir core
    corey = {
        0: Corey(sgrmax = 0.40,Pc_drainage_section = 'Pc_drainage',Pc_imbibition_section = 'Pc_imbibition',nowetting_d="nonwetting_drainage_kr",
                 nowetting_i="nonwetting_imbibition_kr", wetting_type='wetting_kr',nw=2, ng=1.5, swc=0.2, sgc=0.1, krwe=1.0, krge=0.8, labda=2., p_entry=2., pcmax=30, c2=1.5,a=0.8),
        1: Corey(sgrmax = 0.40,Pc_drainage_section = 'Pc_drainage',Pc_imbibition_section = 'Pc_imbibition',nowetting_d="nonwetting_drainage_kr",
                 nowetting_i="nonwetting_imbibition_kr", wetting_type='wetting_kr',nw=2, ng=1.5, swc=0.2, sgc=0.1, krwe=1.0, krge=0.8, labda=2., p_entry=2., pcmax=30, c2=1.5,a=0.8)
    }
    zero = 1e-12
    m.set_physics(corey = corey, zero =zero, n_points=n_points, components=["H2O", "CO2"],temperature=338.15
                  )  # n_point Number of OBL points along axes
    m.inj_stream ={0: [zero],
                   1: [1-zero]}

    if  theraml:
        m.inj_stream[0].append(T_inj)  # Append temperature to CO2
        m.inj_stream[1].append(T_inj)  # Append temperature to H2O
    # m.inj_rate =[0.027,0]  # KMOL/day
    inj_rate = 6.734006734006734e-05*3600*24
    # inj_rate = 1.481481481481482e-05 * 3600 * 24
    m.inj_rate =[inj_rate,0]  # kg/day
    m.p_prod = p_prod
    m.set_sim_params(first_ts=1e-4, mult_ts=1.5, max_ts=365, tol_newton=1e-3, tol_linear=1e-3, it_newton=16,
                     it_linear=20)  # set simulation parameters.
    m.params.linear_type = sim_params.linear_solver_t.cpu_superlu
    m.data_ts.eta[-1] = 0.05
    m.init()
    m.set_output(output_folder = output_folder, all_phase_props= True)
    volume = np.array(m.reservoir.mesh.volume, copy=False)
    poro = np.array(m.reservoir.mesh.poro, copy=False)
    volume_poro = sum(volume * poro)
    print("Pore volume = " + str(sum(volume * poro)))
    ratio = True
    eps = 1e-8
    output_properties = list(m.physics.property_containers[0].output_props.keys()) + list(m.physics.vars)+['Sgmax']
    m.output.output_to_vtk(ith_step = 0,output_properties=output_properties)
    for i in range(100):
        dt = 10
        m.run_hysteresis(dt)
        if continuous:
            if m.physics.engine.t >= 40*dt:
                m.inj_rate[0] = 0
                m.inj_rate[1] = 0
        else:
            if m.physics.engine.t >= 2400:
                m.inj_rate[0] = inj_rate
                m.inj_rate[1] = 0
            elif m.physics.engine.t >= 1800:
                m.inj_rate[0] = 0
                m.inj_rate[1] = inj_rate
            elif m.physics.engine.t >= 1200:
                m.inj_rate[0] = inj_rate
                m.inj_rate[1] = 0
            elif m.physics.engine.t >= 600:
                m.inj_rate[0] = 0
                m.inj_rate[1] = inj_rate
        m.output.output_to_vtk(ith_step = i+1,output_properties=output_properties)
        # elif 200- eps <= m.physics.engine.t < 300 and ratio:
        #     print("Injection I1 stop, and I2 start injecting")
        #     m.reservoir.wells[0].control = m.physics.new_bhp_prod(p_prod)
        #     m.reservoir.wells[2].control = m.physics.new_rate_inj(5, m.inj_stream[1], 0)
        #     # m.reservoir.wells[0].control = m.physics.new_rate_inj(0, m.inj_stream[0], 1)
        #     # m.reservoir.wells[1].control = m.physics.new_rate_inj(5, m.inj_stream[1], 0)
        #     # m.reservoir.wells[3].control = m.physics.new_rate_prod(2,1)
        #     # m.inj_rate = 0
        #     # m.reservoir.wells[1].control = m.physics.new_bhp_inj(1.2, m.inj_stream)
        #     ratio = False
        # elif m.physics.engine.t >= 300:
        #     m.reservoir.wells[2].control = m.physics.new_rate_inj(0, m.inj_stream[1], 0)
m.print_timers()
m.print_stat()
