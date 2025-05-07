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
from darts.engines import well_control_iface

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
    # {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'],
    #     'inj_stream': [1-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'binary', 'post_process': None, 'gpu_device': False},

    {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': False, 'components': ['CO2', 'H2O'],
        'inj_stream': [1-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'binary', 'post_process': None, 'gpu_device': False},

    #{'check_rates': True, 'temperature': None, 'components': ['H2S', 'CO2', 'H2O'], 'inj_stream': [0.05-1e-10, 0.95-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'ternary_H2S_5', 'gpu_device': None},
    #{'check_rates': True, 'temperature': None, 'components': ['C1', 'CO2', 'H2O'], 'inj_stream': [0.05-1e-10, 0.95-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'ternary_C1_5', 'gpu_device': None},
    #{'check_rates': True, 'temperature': None, 'components': ['C1', 'H2S', 'CO2', 'H2O'], 'inj_stream': [0.04, 0.01, 0.95, 283.15], 'nx': nx, 'nz': nz, 'output_dir': '4components_5', 'gpu_device': None}
    ]

# for specs in model_specs:
for specs in model_specs:
    print(specs)

    # print(os.environ["CONDA_PREFIX"])
    output_dir = specs['output_dir'] if specs['post_process'] is None else os.path.join(specs['output_dir'], specs['post_process'])
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
        m.inj_rate = [inj_rate, inj_rate]  # [well 1, well 2]

    if specs['RHS'] is False:
        m.set_well_controls()

    """ Init model """
    # now that the reservoir and physics is defined, you can init the DartsModel()
    if specs['gpu_device'] is False:
        m.platform = 'cpu'
    else:
        m.platform = 'gpu'
        set_gpu_device(0)

    if specs['post_process'] is None:
        m.init(discr_type='tpfa', platform=m.platform)
        m.set_output(output_folder=output_dir, verbose = True)
    else:
        print(f'Post processing into {output_dir}...')
        m.init(discr_type='tpfa', platform=m.platform)
        m.set_output(output_folder=output_dir, save_initial=False)

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

    if specs['check_rates'] and specs['post_process'] is None:
        time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=0)
        m.output.save_property_array(time_vector, property_array, 'property_array_ts0.h5')

        mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
        mass_components_n = {key: np.sum(value) for key, value in mass_per_component.items()} # sum over grid blocks per component
        avg_rates = []

    vtk_array = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900,
         1000]) # years for which to export a .vtk file

    m.inj_rate = [3024, 0]
    event1 = True # turning on well 2
    event2 = True # turning off wells

    if specs['post_process'] is None:
        Nt = 10
    else:
        Nt = 0 # skip simulation
    Dt = 365/10

    timing = []
    start = time.time()
    for ts in range(Nt):
        # m.run(Dt, verbose=True)  # run model for 1 year
        m.run_python_my(Dt, restart_dt=Dt/10)

        if m.physics.engine.t >= 1 * 36.5 and m.physics.engine.t < 50 * 365:
            # At 25 years, start injecting in the second well
            m.inj_rate = [inj_rate, inj_rate]
            m.physics.set_well_controls(well=m.reservoir.wells[1],
                                        is_control=True,
                                        control_type=well_control_iface.MASS_RATE,
                                        is_inj=True,
                                        target=m.inj_rate[1],
                                        phase_name='V',
                                        inj_composition=m.inj_stream[:-1],
                                        inj_temp=283.15)

        elif m.physics.engine.t >= 10 * 365 and event1:
            # At 50 years, stop injection for both wells
            m.inj_rate = [0, 0]
            specs['check_rates'] = False # after injection stop checking rates
            event1 = False
        else:
            pass

        if specs['check_rates']:
            time_vector, property_array = m.output.output_properties(output_properties=output_props, engine=True)
            mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
            mass_components = {key: np.sum(value) for key, value in mass_per_component.items()}

            for i, name in enumerate(m.components):
                rate = (mass_components[name] - mass_components_n[name]) / Dt
                avg_rates.append(rate)
                print(f'Injecting {name} at {avg_rates[i::m.nc][0]} kg/day.')
            print('----------------------------------------------------------------')

            mass_components_n = mass_components
        else:
            pass

        if m.physics.engine.t/Dt in vtk_array:
            # save data
            m.output.save_data_to_h5('reservoir')

            # save property_array
            time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=ts+1)
            m.output.save_property_array(time_vector, property_array, f'property_array_ts{ts+1}.h5')

            # plt plot
            for i, name in enumerate(property_array.keys()):
                plt.figure(figsize=(10, 2))
                plt.title(f'{name} @ year {time_vector[0]}')
                c = plt.pcolor(grid[0], grid[1], property_array[name][0].reshape(nz, nx), cmap='cividis')
                plt.colorbar(c, aspect=10)
                plt.xlabel('x [m]'); plt.ylabel('z [m]')
                plt.savefig(os.path.join(m.output_folder, 'figures', f'{name}_ts_{ts+1}.png'), bbox_inches='tight')
                plt.close()

            mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
            for i, name in enumerate(mass_per_component.keys()):
                plt.figure(figsize=(10, 2))
                plt.title(f'{name} mass @ year {time_vector[0]}')
                c = plt.pcolor(grid[0], grid[1], mass_per_component[name].reshape(nz, nx)/1e6, cmap='jet')
                plt.colorbar(c, label = 'kt', aspect=10)
                plt.xlabel('x [m]'); plt.ylabel('z [m]')
                plt.savefig(os.path.join(m.output_folder, 'figures', f'{name}_mass_ts_{ts+1}.png'), bbox_inches='tight')
                plt.close()

            # output to .vtk
            m.output.output_to_vtk(output_properties=output_props, ith_step=ts+1)
        else:
            pass

        m.print_timers()

    stop = time.time()
    print("Runtime = %3.2f sec" % (stop - start))

    if specs['check_rates']:
        avg_rates = np.array(avg_rates)

        # # Define target rates
        # target_rates = np.ones(Nt)
        # target_rates[time_vector < 25*365] *= 3024
        # target_rates[time_vector > 25*365 and time_vector < 50*365] *= 2 * 3024
        # target_rates[time_vector >= 50*365] = 0
        #
        # # Compute percentage deviation, avoiding division by zero
        # percentage_deviation = np.where(target_rates != 0,
        #                                 ((avg_rates - target_rates) / target_rates) * 100,
        #                                 np.nan)

        for i, name in enumerate(m.components):
            fig, ax1 = plt.subplots(dpi=100)
            ax1.grid()
            ax1.step(avg_rates[i::m.nc], 'b-o', label="Avg Rates")
            # ax1.axhline(y=3024, color='k', linestyle='--', label='3024 m³/day')
            # ax1.axhline(y=2*3024, color='k', linestyle='--', label='2 * 3024 m³/day')
            ax1.set_xlabel("Time (years)")
            ax1.set_ylabel("Injection Rate (m³/day)", color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            # ax2 = ax1.twinx()
            # ax2.step(time_steps, percentage_deviation, 'r--s', label="Percentage Deviation (%)")
            # ax2.set_ylabel("Deviation (%)", color='r')
            # ax2.tick_params(axis='y', labelcolor='r')
            # ax1.legend()
            # plt.title("Injection Rates & Percentage Deviation")
            # plt.savefig(os.path.join(m.output_folder, 'rates.png'), bbox_inches='tight')
            plt.show()