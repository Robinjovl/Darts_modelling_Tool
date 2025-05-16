"""
    Coarse-scale, isothermal, SPE11b
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

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

def plt_plot(dictionary, filename):
    grid = np.meshgrid(np.linspace(0, 8400, nx), np.linspace(0, 1200, nz))

    for i, name in enumerate(dictionary.keys()):
        plt.figure(figsize=(10, 2))
        plt.title(f'{name} @ year {time_vector[0]}')
        c = plt.pcolor(grid[0], grid[1], dictionary[name][0].reshape(nz, nx), cmap='cividis')
        plt.colorbar(c, aspect=10)
        plt.xlabel('x [m]');
        plt.ylabel('z [m]')
        plt.savefig(os.path.join(m.output_folder, 'figures', filename), bbox_inches='tight')
        plt.close()

    return 0

def set_well_rhs(m, Dt, inj_rate, event1, event2):
    if m.physics.engine.t >= 25 * Dt and m.physics.engine.t < 50 * Dt and event1:
        print('At 25 years, start injecting in the second well')
        m.inj_rate = [inj_rate, inj_rate]
        event1 = False
    elif m.physics.engine.t >= 50 * Dt and event2:
        print('At 50 years, stop injection for both wells')
        m.inj_rate = [m.zero, m.zero]
        specs['check_rates'] = False  # after injection stop checking rates
        event2 = False

    return event1, event2

def set_well_rates(m, Dt, inj_rate, event1, event2):
    if m.physics.engine.t >= 25 * Dt and m.physics.engine.t < 50 * Dt and event1:
        print('At 25 years, start injecting in the second well')
        m.inj_rate = [inj_rate, inj_rate]
        m.physics.set_well_controls(wctrl = m.reservoir.wells[1].control,
                                    control_type=well_control_iface.MASS_RATE, 
                                    is_inj = True,
                                    target = m.inj_rate[1], 
                                    phase_name = 'V',
                                    inj_composition = m.inj_stream[:-1],
                                    inj_temp = m.inj_stream[-1]
                                    )
        
        event1 = False

    elif m.physics.engine.t >= 50 * Dt and event2:
        print('At 50 years, stop injection in both wells')
        m.inj_rate = [m.zero, m.zero]
        for i in range(2):
            m.physics.set_well_controls(wctrl = m.reservoir.wells[i].control,
                                        control_type = well_control_iface.MASS_RATE,
                                        is_inj = True,
                                        target = m.inj_rate[i],
                                        phase_name = 'V',
                                        inj_composition = m.inj_stream[:-1],
                                        inj_temp = m.inj_stream[-1]
                                        )
        specs['check_rates'] = False  # after injection stop checking rates
        event2 = False

    return event1, event2

def build_output_dir(spec, base_dir="results"):
    iso_tag = "iso" if spec.get("temperature") is not None else "niso"
    rhs_tag = "rhs" if spec.get("RHS") else "wells"
    disp_tag = "disp" if spec.get("dispersion") else "nodisp"
    components = "-".join(spec.get("components", []))
    nx = spec.get("nx", "nx?")
    nz = spec.get("nz", "nz?")
    device = 'CPU' if spec.get('gpu_device') is False else 'GPU'
    
    # Construct folder name
    dir_name = f"{iso_tag}__{rhs_tag}__{disp_tag}__{components}__nx{nx}_nz{nz}_{device}"
    return os.path.join(base_dir, dir_name)

def run(specs):
    """ set up output directory """
    if specs['output_dir'] is None:
        specs["output_dir"] = build_output_dir(specs)
    print(specs)
    # print(os.environ["CONDA_PREFIX"])
    
    output_dir = specs['output_dir'] if specs['post_process'] is None else os.path.join(specs['output_dir'], specs['post_process'])
    os.makedirs(output_dir, exist_ok=True)

    # save specs to a .pkl file
    with open(os.path.join(output_dir, 'specs.pkl'), 'wb') as f:
        pickle.dump(specs, f)

    redirect_darts_output(os.path.join(output_dir, 'model.log'))
    m = Model(specs)

    """Define physics"""
    # zero = 1e-10
    m.set_physics(zero=zero, temperature=specs['temperature'], n_points=1001)

    # solver paramters
    m.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=365, tol_linear=1e-2, tol_newton=1e-3,
                     it_linear=50, it_newton=12, newton_type=sim_params.newton_global_chop)
    m.params.newton_params[0] = 0.05*2
    m.data_ts.eta=np.ones(m.physics.n_vars)
    
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
    m.inj_rate = [inj_rate, zero] if specs['1000years'] is False else [0, 0]

    """ Init model """
    # now that the reservoir and physics is defined, you can init the DartsModel()
    if specs['gpu_device'] is False:
        m.platform = 'cpu'
    else:
        m.platform = 'gpu'
        set_gpu_device(0)

    if specs['post_process'] is None:
        m.init(discr_type='tpfa', platform=m.platform)
        m.set_output(output_folder=output_dir, sol_filename='single_reservoir_solution.h5', save_initial = not specs['1000years'], precision='s', verbose = True)
    else:
        print(f'Post processing into {output_dir}...')
        m.init(discr_type='tpfa', platform=m.platform)
        m.set_output(output_folder=output_dir, save_initial=False)
        
    # simulate a thousand years 
    if specs['1000years']:
        for i in range(100):
            print(f'-------- Year {i}/100 --------')
            m.run(365, restart_dt=365/2, save_reservoir_data=False, save_well_data=False)
        m.physics.engine.t = 0.
        m.output.save_data_to_h5(kind = 'reservoir')
        m.inj_rate = [inj_rate, 0]
    m.output.verbose = False
    
    if specs['dispersion']:
        m.init_dispersion()        

    if 1:
        nx = m.reservoir.nx
        nz = m.reservoir.nz
        nb = m.reservoir.n
        n_vars = m.physics.n_vars
        vars = m.physics.vars
        grid = np.meshgrid(np.linspace(0, 8400, nx),
                           np.linspace(0, 1200, nz))
        poro = m.reservoir.global_data['poro']
        op_num = np.array(m.reservoir.mesh.op_num)[:m.reservoir.n] + 1

        plt.figure(dpi = 100, figsize = (10, 2))
        plt.title('Facies')
        c = plt.pcolor(grid[0], grid[1], op_num.reshape(nz, nx), cmap = 'jet', vmin=min(op_num), vmax=max(op_num))
        plt.colorbar(c, ticks = np.arange(1, 8))
        plt.xlabel('x [m]'); plt.ylabel('z [m]')
        # centroids = m.reservoir.discretizer.centroids_all_cells
        centroids = m.reservoir.centroids
        plt.scatter(centroids[m.reservoir.well_cells[0], 0], centroids[m.reservoir.well_cells[0], 2], marker='x', c='r', s=5)
        plt.scatter(centroids[m.reservoir.well_cells[1], 0], centroids[m.reservoir.well_cells[1], 2], marker='x', c='r', s=5)
        plt.savefig(os.path.join(m.output_folder, f'op_num.png'), bbox_inches='tight')
        plt.close()

        solution_vector = np.array(m.physics.engine.X)
        for i, name in enumerate(vars):
            plt.figure(figsize = (10, 2))
            plt.title(name)
            c = plt.pcolor(grid[0], grid[1], np.round(solution_vector[i::n_vars][:nb], 2).reshape(nz, nx), cmap = 'jet')
            plt.colorbar(c, aspect = 10)
            plt.xlabel('x [m]'); plt.ylabel('z [m]')
            plt.savefig(os.path.join(m.output_folder, f'initial_conditions_{name}.png'), bbox_inches='tight')
            plt.close()

    output_props = m.physics.vars + m.output.properties
    m.output.output_to_vtk(output_properties=output_props, ith_step=0)

    avg_rates = []
    if specs['check_rates'] and specs['post_process'] is None:
        time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=0)
        m.output.save_property_array(time_vector, property_array, 'property_array_ts0.h5')
        mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
        mass_components_n = {key: np.sum(value) for key, value in mass_per_component.items()} # sum over grid blocks per component

    vtk_array = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) # years for which to export a .vtk file

    event1 = True
    event2 = True

    if specs['post_process'] is None:
        Nt = 1
    else:
        Nt = 0 # skip simulation
    Dt = 365

    for ts in range(Nt):
        print(f'----------------------------------- Simulate from year {(ts*Dt)/365} until year {((ts+1)*Dt)/365} -----------------------------------')
        m.run(Dt, restart_dt = 1.0, save_reservoir_data = False, save_well_data = not specs['RHS'], verbose=True)

        if specs['check_rates']:
            time_vector, property_array = m.output.output_properties(output_properties=output_props, engine=True)
            mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
            mass_components = {key: np.sum(value) for key, value in mass_per_component.items()}

            for i, name in enumerate(m.components):
                rate = (mass_components[name] - mass_components_n[name]) / Dt
                avg_rates.append(rate)
                print(f'Injecting {name} at {avg_rates[i::m.nc][-1]} kg/day.')
            mass_components_n = mass_components
            
        else:
            pass

        if m.physics.engine.t/Dt in vtk_array:
            # save data
            m.output.save_data_to_h5('reservoir')

            # save property_array
            time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=-1)
            m.output.save_property_array(time_vector, property_array, f'property_array_ts{ts+1}.h5')

            for i, name in enumerate(property_array.keys()):
                plt.figure(figsize=(10, 2))
                plt.title(f'{name} @ year {time_vector[0]/365}')
                c = plt.pcolor(grid[0], grid[1], property_array[name][0].reshape(nz, nx), cmap='cividis')
                plt.colorbar(c, aspect=10)
                plt.xlabel('x [m]'); plt.ylabel('z [m]')
                plt.savefig(os.path.join(m.output_folder, 'figures', f'{name}_ts_{ts+1}.png'), bbox_inches='tight')
                plt.close()

            mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
            for i, name in enumerate(mass_per_component.keys()):
                plt.figure(figsize=(10, 2))
                plt.title(f'{name} mass @ year {time_vector[0]/365}')
                c = plt.pcolor(grid[0], grid[1], mass_per_component[name].reshape(nz, nx)/1e6, cmap='cividis')
                plt.colorbar(c, label = 'kt', aspect=10)
                plt.xlabel('x [m]'); plt.ylabel('z [m]')
                plt.savefig(os.path.join(m.output_folder, 'figures', f'{name}_mass_ts_{ts+1}.png'), bbox_inches='tight')
                plt.close()
                
            if specs['dispersion']:
                darcy_velocities = np.asarray(m.physics.engine.darcy_velocities).reshape(m.reservoir.mesh.n_res_blocks, m.physics.nph, 3)
                for p, ph in enumerate(m.physics.phases):
                    for v, orientation in enumerate(['x', 'y', 'z']):
                        plt.figure(figsize=(10, 2))
                        plt.title(f'Velocity of {ph} in {orientation} direction @ year {time_vector[0]/365}')
                        c = plt.pcolor(grid[0], grid[1], darcy_velocities[:, p, v].reshape(nz, nx), cmap='cividis')
                        plt.colorbar(c, aspect=10)
                        plt.xlabel('x [m]'); plt.ylabel('z [m]')
                        plt.savefig(os.path.join(m.output_folder, 'figures', f'vel_{ph}_{orientation}_{ts+1}.png'), bbox_inches='tight')
                        plt.close()
                        
        else:
            pass

        if specs['RHS']:
            event1, event2 = set_well_rhs(m, Dt, inj_rate, event1, event2)
        else:
            event1, event2 = set_well_rates(m, Dt, inj_rate, event1, event2)

    return m, avg_rates

#%%

"""Define realization ID"""
nx = 840
nz = 120
zero = 1e-10
model_specs = [
    # # BINARY ISOTHERMAL MODEL
    #     # RHS CORRECTION WITH DISPERSION OFF
    # {'check_rates': True, 'temperature': 273.15+40, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'], 'inj_stream': [1-1e-10, 283.15],
    #       'nx': nx, 'nz': nz, 'dispersion': False, 'output_dir': None, 'post_process': None, 'gpu_device': False},
    #
    #     # RHS CORRECTION WITH DISPERSION ON
    # {'check_rates': True, 'temperature': 273.15+40, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'], 'inj_stream': [1-1e-10, 283.15],
    #       'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': None, 'post_process': None, 'gpu_device': False},
    #
    #     # WELLS WITH DISPERSION OFF
    # {'check_rates': True, 'temperature': 273.15+40, '1000years': False, 'RHS': False, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
    #     'nx': nx, 'nz': nz, 'dispersion': False, 'output_dir': None, 'post_process': None, 'gpu_device': False},
    #
    #     # WELLS WITH DISPERSION ON
    # {'check_rates': True, 'temperature': 273.15+40, '1000years': False, 'RHS': False, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
    #     'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': None, 'post_process': None, 'gpu_device': False},

    # BINARY NON-ISOTHERMAL MODEL
        # RHS CORRECTION WITH DISPERSION OFF
    # {'check_rates': True, 'temperature': None, '1000years': True, 'RHS': True, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
    #     'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': None, 'post_process': None, 'gpu_device': False},
         
    #     # RHS CORRECTION WITH DISPERSION ON
    # {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
    #     'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': None, 'post_process': None, 'gpu_device': False},
    #
        # WELLS WITH DISPERSION OFF
    {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
        'nx': nx, 'nz': nz, 'dispersion': False, 'output_dir': None, 'post_process': None, 'gpu_device': False},
    #
    #     # WELLS WITH DISPERSION ON
    # {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': False, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
    #     'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': None, 'post_process': None, 'gpu_device': False},
    
    #{'check_rates': True, 'temperature': None, 'components': ['H2S', 'CO2', 'H2O'], 'inj_stream': [0.05-1e-10, 0.95-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'ternary_H2S_5', 'gpu_device': None},
    #{'check_rates': True, 'temperature': None, 'components': ['C1', 'CO2', 'H2O'], 'inj_stream': [0.05-1e-10, 0.95-1e-10, 283.15], 'nx': nx, 'nz': nz, 'output_dir': 'ternary_C1_5', 'gpu_device': None},
    #{'check_rates': True, 'temperature': None, 'components': ['C1', 'H2S', 'CO2', 'H2O'], 'inj_stream': [0.04, 0.01, 0.95, 283.15], 'nx': nx, 'nz': nz, 'output_dir': '4components_5', 'gpu_device': None}
    ]
    

#%%

if __name__ == '__main__':
    for specs in model_specs:
        m, avg_rates = run(specs)
        m.print_timers()
        m.output.output_to_vtk(output_properties = m.physics.vars + m.output.properties)

        if specs['RHS'] is False:
            time_data = m.output.store_well_time_data(["components_mass_rates"])
            m.output.plot_well_time_data(["components_mass_rates"])

        if specs['check_rates']:
            avg_rates = np.array(avg_rates)
            for i, name in enumerate(m.components):
                fig, ax1 = plt.subplots(dpi=100)
                ax1.grid()
                ax1.step(avg_rates[i::m.nc], 'b-o', label="Avg Rates")
                ax1.set_xlabel("Time (years)")
                ax1.set_ylabel("Injection Rate (m³/day)", color='k')
                ax1.tick_params(axis='y', labelcolor='k')
                plt.savefig(os.path.join(m.output_folder, f'rate_check_{name}.png'))
                plt.close()