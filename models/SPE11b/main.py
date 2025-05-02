"""
    Coarse-scale, isothermal, SPE11b
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import h5py
import time

from darts.reservoirs.mesh.geometry.map_mesh import MapMesh, _translate_curvature
from model_b import Model, PorPerm, Corey, layer_props
from darts.engines import redirect_darts_output, sim_params
try:
    from darts.engines import set_gpu_device
except ImportError:
    pass
from fluidflower_str_b import FluidFlowerStruct

#%%

# For each of the facies within the SPE11b model we define a set of operators in the physics.
property_regions  = [0, 1, 2, 3, 4, 5, 6]
layers_to_regions = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}

"""Define realization ID"""
nx = 840//4
nz = 120//4
model_specs = [
    {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'],
        'inj_stream': [1-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'binary', 'gpu_device': False},

    #{'check_rates': True, 'temperature': None, 'components': ['H2S', 'CO2', 'H2O'], 'inj_stream': [0.05-1e-10, 0.95-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'ternary_H2S_5', 'gpu_device': None},
    #{'check_rates': True, 'temperature': None, 'components': ['C1', 'CO2', 'H2O'], 'inj_stream': [0.05-1e-10, 0.95-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'ternary_C1_5', 'gpu_device': None},
    #{'check_rates': True, 'temperature': None, 'components': ['C1', 'H2S', 'CO2', 'H2O'], 'inj_stream': [0.04, 0.01, 0.95, 283.15], 'nx': nx, 'nz': nz, 'output_dir': '4components_5', 'gpu_device': None}

    ]

# for specs in model_specs:
for specs in model_specs:
    print(specs)

    # print(os.environ["CONDA_PREFIX"])
    output_dir = specs['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # save specs to a pkl
    with open(os.path.join(output_dir, 'specs.pkl'), 'wb') as f:
        pickle.dump(specs, f)

    redirect_darts_output(os.path.join(output_dir, 'model.log'))
    m = Model(specs)

    """Define physics"""
    zero = 1e-10
    m.set_physics(zero=zero, temperature=specs['temperature'], n_points=1001, diff=1e-9)

    # solver paramters
    m.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=365, tol_linear=1e-3, tol_newton=1e-3,
                     it_linear=50, it_newton=12, newton_type=sim_params.newton_global_chop)
    m.params.newton_params[0] = 0.05
    m.params.nonlinear_norm_type = m.params.LINF # linf if you use m.set_rhs() for injection

    """Define the reservoir and wells """
    well_centers = {
        "I1": [2700.0, 0.0, 300.0],
        "I2": [5100.0, 0.0, 700.0]
    }

    # structured reservoir
    m.reservoir = FluidFlowerStruct(timer=m.timer, layer_properties=layer_props, layers_to_regions=layers_to_regions,
                                    model_specs=specs, well_centers=well_centers)
    m.set_str_boundary_volume_multiplier()  # right and left boundary volume multiplier

    """ Define initial and boundary conditions """
    # define injection stream of the wells
    m.inj_stream = specs['inj_stream']
    inj_rate = 3024 # mass rate per well, kg/day

    if specs['1000years']:
        m.inj_rate = [0, 0]
        m.run(1000)
    else:
        m.inj_rate = [inj_rate, 0]  # [well 1, well 2]

    if specs['RHS'] is False:
        m.set_well_controls()

    """ Init model """
    # now that the reservoir and physics is defined, you can init the DartsModel()
    if specs['gpu_device'] is False:
        m.platform = 'cpu'
    else:
        m.platform = 'gpu'
        set_gpu_device(0)
    m.init(discr_type='tpfa', platform=m.platform, restart = False)
    m.set_output(output_folder=output_dir, verbose = True)

    if 1:
        nx = m.reservoir.nx
        nz = m.reservoir.nz
        nb = m.reservoir.n
        n_vars = m.physics.n_vars
        vars = m.physics.vars
        grid = np.meshgrid(np.linspace(0, 8400, nx),
                           np.linspace(0, 1200, nz))
        poro = m.reservoir.global_data['poro']

        plt.figure(dpi = 100, figsize = (10, 2))
        plt.title('Porosity')
        c = plt.pcolor(grid[0], grid[1], poro.reshape(nz, nx))
        plt.colorbar(c)
        plt.xlabel('x [m]'); plt.ylabel('z [m]')
        plt.savefig(os.path.join(m.output_folder, f'porosity.png'), bbox_inches='tight')
        plt.close()

        solution_vector = np.array(m.physics.engine.X)
        for i, name in enumerate(vars):
            plt.figure(figsize = (10, 2))
            plt.title(name)
            c = plt.pcolor(grid[0], grid[1], solution_vector[i::n_vars][:nb].reshape(nz, nx), cmap = 'jet')
            plt.colorbar(c, aspect = 10)
            plt.xlabel('x [m]'); plt.ylabel('z [m]')
            plt.savefig(os.path.join(m.output_folder, f'initial_conditions_{name}.png'), bbox_inches='tight')
            plt.close()

    output_props = m.physics.vars + m.output.properties
    m.output.output_to_vtk(output_properties=output_props, ith_step=0)

    if specs['check_rates']:
        time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=0)
        m.output.save_property_array(time_vector, property_array, 'property_array_ts0.h5')

        mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
        mass_components_n = {key: np.sum(value) for key, value in mass_per_component.items()}
        avg_rates = []

    vtk_array = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900,
         1000]) # years for which to export a .vtk file

    m.inj_rate = [3024, 0]
    event1 = True # turning on well 2
    event2 = True # turning off wells

    Nt = 1
    Dt = 365

    timing = []
    start = time.time()

    for i in range(Nt):
        if 0:
            m.run(Dt, verbose=True)  # run model for 1 year
        else:
            m.run_python_my(Dt)
            m.output.save_data_to_h5('solution')

        if m.physics.engine.t >= 25 * 365 and m.physics.engine.t < 50 * 365:
            # At 25 years, start injecting in the second well
            m.inj_rate = [inj_rate, inj_rate]
            m.set_well_controls()

        elif m.physics.engine.t >= 50 * 365 and event1:
            # At 50 years, stop injection for both wells
            m.inj_rate = [0, 0]
            m.set_well_controls()
            specs['check_rates'] = False
            event1 = False

        if specs['check_rates']:
            property_array = m.output_properties_old()
            mass_co2 = m.get_mass_components(property_array)

            for i, name in enumerate(m.components):
                rate = (mass_co2[name] - mass_co2_n[name]) / (5 * 365)
                avg_rates.append(rate)
                print(f'Injecting {name} at {rate} kg/day.')
            print('----------------------------------------------------------------')

            mass_co2_n = mass_co2

    stop = time.time()
    print("Runtime = %3.2f sec" % (stop - start))

    #%%

    nx, nz = m.reservoir.nx, m.reservoir.nz  # grid dimensions
    prop_list = list(m.physics.property_containers[0].output_props.keys())

    M_H2O = m.physics.property_containers[0].Mw[0] # molar mass water in kg/kmol
    M_CO2 = m.physics.property_containers[0].Mw[1] # molar mass CO2 in kg/kmol
    PV = np.array(m.reservoir.mesh.volume)[1] * np.array(m.reservoir.mesh.poro) # pore volume
    PV = PV[:m.reservoir.n]
    # Generate some example data to animate
    time_vector, property_array = m.output_properties(output_properties = prop_list, timestep = None)
    data = []
    for i in range(len(time_vector)):
        # vapour mass fraction
        wco2 = property_array['yCO2'][i] * M_CO2 / (property_array['yCO2'][i] * M_CO2 + (1 - property_array['yCO2'][i]) * M_H2O)

        # mass of CO2 in the aqueous phase
        mass_co2_aq = PV * (1 - property_array['satV'][i]) * property_array['xCO2'][i] * property_array['rho_mA'][i] * M_CO2

        # mass of CO2 in the vapor phase
        mass_co2_v = PV * property_array['satV'][i] * property_array['rhoV'][i] * wco2

        # total mass of CO2 in kton
        mass_co2 = (mass_co2_aq + mass_co2_v)/1e6

        # append to data list for plotting
        data.append(mass_co2.reshape(m.reservoir.nz, m.reservoir.nx))

    plt.figure(dpi = 100, figsize = (10, 2))
    plt.title('Mass of CO$_2$ in kt')
    c = plt.pcolor(grid[0], grid[1], data[-1], cmap = 'jet')
    plt.colorbar(c, aspect = 10)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    # plt.tight_layout()
    plt.savefig(os.path.join(m.output_folder, f'year_{int(time_vector[-1]/365)}_mass_co2.png'), bbox_inches='tight')
    plt.close()

    #%%

    solution_vector = np.array(m.physics.engine.X)

    for i, name in enumerate(m.physics.vars):

        plt.figure(dpi = 100, figsize = (10, 2))
        plt.title(name + f' @ year {int(time_vector[-1]/365)}')
        c = plt.pcolor(grid[0],
                       grid[1],
                       solution_vector[i::m.physics.n_vars][:m.reservoir.n].reshape(m.reservoir.nz, m.reservoir.nx),
                       cmap = 'jet')
        plt.colorbar(c, aspect = 10)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.savefig(os.path.join(m.output_folder, f'year_{int(time_vector[-1]/365)}_{name}.png'), bbox_inches='tight')
        plt.show()



    #%%

    if specs['check_rates']:
        avg_rates = np.array(avg_rates)

        # Define target rates
        target_rates = np.ones(Nt)
        target_rates[time_vector < 25*365] *= 3024
        target_rates[time_vector > 25*365 and time_vector < 50*365] *= 2 * 3024
        target_rates[time_vector >= 50*365] = 0

        # Compute percentage deviation, avoiding division by zero
        percentage_deviation = np.where(target_rates != 0,
                                        ((avg_rates - target_rates) / target_rates) * 100,
                                        np.nan)

        # Create figure and primary y-axis
        fig, ax1 = plt.subplots(dpi=100)
        ax1.grid()

        # Plot avg_rates on the primary y-axis
        ax1.step(time_steps, avg_rates, 'b-o', label="Avg Rates")
        ax1.axhline(y=3024, color='k', linestyle='--', label='3024 m³/day')
        ax1.axhline(y=2*3024, color='k', linestyle='--', label='2 * 3024 m³/day')

        # Configure primary y-axis
        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Injection Rate (m³/day)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create secondary y-axis
        ax2 = ax1.twinx()
        ax2.step(time_steps, percentage_deviation, 'r--s', label="Percentage Deviation (%)")

        # Configure secondary y-axis
        ax2.set_ylabel("Deviation (%)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Legends
        ax1.legend()
        # ax2.legend()

        plt.title("Injection Rates & Percentage Deviation")
        plt.savefig(os.path.join(m.output_folder, 'rates.png'), bbox_inches='tight')
        plt.show()