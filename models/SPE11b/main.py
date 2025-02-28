"""

Coarse-scale, isothermal, SPE11b

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import h5py

from darts.reservoirs.mesh.geometry.map_mesh import MapMesh, _translate_curvature
from model_b import Model, PorPerm, Corey, layer_props
from darts.engines import redirect_darts_output, sim_params
from fluidflower_str_b import FluidFlowerStruct

# For each of the facies within the SPE11b model we define a set of operators in the physics.
property_regions  = [0, 1, 2, 3, 4, 5, 6]
layers_to_regions = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}

# define the Corey parameters for each layer (rock type) according to the technical description of the CSP
corey = {
    0: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1.935314, pcmax=300, c2=1.5),
    1: Corey(nw=1.5, ng=1.5, swc=0.14, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.08655, pcmax=300, c2=1.5),
    2: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0612, pcmax=300, c2=1.5),
    3: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.038706, pcmax=300, c2=1.5),
    4: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0306, pcmax=300, c2=1.5),
    5: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300, c2=1.5),
    6: Corey(nw=1.5, ng=1.5, swc=1e-8, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1e-2, pcmax=300, c2=1.5)
}



"""Define realization ID"""
model_specs = [
    {'check_rates': True, 'temperature': None, 'components': ['CO2', 'H2O'], 'inj_stream': [1-1e-10, 283.15], 'nx': 170//2, 'nz': 60//2, 'output_dir': 'binary'},
    {'check_rates': True, 'temperature': None, 'components': ['H2S', 'CO2', 'H2O'], 'inj_stream': [0.01-1e-10, 0.99-1e-10, 283.15], 'nx': 170//2, 'nz': 60//2, 'output_dir': 'ternary_H2S'},
    {'check_rates': True, 'temperature': None, 'components': ['C1' , 'CO2', 'H2O'], 'inj_stream': [0.01-1e-10, 0.99-1e-10, 283.15], 'nx': 170//2, 'nz': 60//2, 'output_dir': 'ternary_C1'},
    {'check_rates': True, 'temperature': None, 'components': ['C1' , 'H2S', 'CO2', 'H2O'], 'inj_stream': [0.04-1e10, 0.01-1e10, 0.95-1e-10, 283.15], 'nx': 170//2, 'nz': 60//2, 'output_dir': '4components'}
    ]

# for specs in model_specs:
for j in [3]:
    specs = model_specs[j]
    
    os.makedirs(specs['output_dir'], exist_ok=True)
    
    # Save to a file
    with open(os.path.join(specs['output_dir'], 'specs.pkl'), 'wb') as f:
        pickle.dump(specs, f)
    
    redirect_darts_output(os.path.join(specs['output_dir'], 'model.log'))
    
    m = Model(specs)

    """Define physics"""
    zero = 1e-10
    m.set_physics(corey=corey, zero=zero, temperature=specs['temperature'], n_points=1001, diff=1e-9)
    
    # solver paramters
    m.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=365, tol_linear=1e-3, tol_newton=1e-3,
                     it_linear=50, it_newton=12, newton_type=sim_params.newton_global_chop)
    m.params.newton_params[0] = 0.05
    m.params.nonlinear_norm_type = m.params.LINF
    
    """Define the reservoir and wells """
    well_centers = {
        "I1": [2700.0, 0.0, 300.0],
        "I2": [5100.0, 0.0, 700.0]
    }
    
    # structured = specs['structured']
    m.reservoir = FluidFlowerStruct(timer=m.timer, layer_properties=layer_props, layers_to_regions=layers_to_regions,
                                    model_specs=specs, well_centers=well_centers) # structured reservoir
    
    if 1:
        grid = np.meshgrid(np.linspace(0, 8400, m.reservoir.nx), np.linspace(0, 1200, m.reservoir.nz))
        plt.figure(figsize = (10, 2))
        plt.title('Porosity')
        c = plt.pcolor(grid[0], grid[1], m.reservoir.global_data['poro'].reshape(m.reservoir.nz, m.reservoir.nx))
        plt.colorbar(c)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.close()
    
    """ Define initial and boundary conditions """
    # define injection stream of the wells
    m.inj_stream = specs['inj_stream']
        
    inj_rate = 3024 # mass rate per well, kg/day
    m.inj_rate = [inj_rate, 0]  # [well 1, well 2]
    
    m.set_str_boundary_volume_multiplier()  # right and left boundary volume multiplier
    
    # now that your reservoir and physics is defined, you can init your DartsModel()
    output_dir = specs['output_dir']
    m.platform = 'cpu'
    m.init(discr_type='tpfa', platform=m.platform, output_folder=output_dir, restart = False)
    
    # equillibration step
    # m.run(365/10, verbose = True)
    # m.run_python_my(365/10)
    # m.physics.engine.t = 0 # return engine time to zero
    
    Nt = 100 # 100 years with output every 5 years
    DT = 365
    import time

    solution_vector = np.array(m.physics.engine.X)
    
    for i, name in enumerate(m.physics.vars):
        plt.figure(figsize = (10, 2))
        plt.title(name)
        c = plt.pcolor(grid[0],
                       grid[1], 
                       solution_vector[i::m.physics.n_vars][:m.reservoir.n].reshape(m.reservoir.nz, m.reservoir.nx),
                       cmap = 'jet')
        plt.colorbar(c, aspect = 10)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.savefig(os.path.join(m.output_folder, f'initial_conditions_{name}.png'), bbox_inches='tight')
        plt.close()

    if specs['check_rates']:
        property_array = m.output_properties_old()
        mass_components_n = {key: np.sum(value) for key, value in m.get_mass_components(property_array).items()}
        avg_rates = []
    
    start = time.time()
    
    m.inj_rate = [3024, 0]
    event1 = True
    event2 = True
    
    for i in range(Nt):
        print(f'-------------------- Simulate from {m.physics.engine.t/365} to {(m.physics.engine.t+365)/365} and set rate to {m.inj_rate} --------------------')
        m.run_python_my(DT, verbose = 1, restart_dt=1)
        m.save_data_to_h5('solution')
        
        if m.physics.engine.t >= 50 * 365 and m.physics.engine.t < 50 * 365 and event1:
            print('At 25 years, start injecting in the second well')
            m.inj_rate = [inj_rate, inj_rate]
            # m.set_well_controls()
            event1 = False
    
        elif m.physics.engine.t >= 50 * 365 and event2:
            print('At 50 years, stop injection for both wells')
            m.inj_rate = [0, 0]
            # m.set_well_controls()
            specs['check_rates'] = False
            event2 = False
            
        # elif m.physics.engine.t < 25 * 365 and event1 and event2:
        #     m.inj_rate = [3024, 0]
        
        if specs['check_rates']:
            property_array = m.output_properties_old()
            mass_components = {key: np.sum(value) for key, value in m.get_mass_components(property_array).items()}
            
            for i, name in enumerate(m.components):
                rate = (mass_components[name] - mass_components_n[name]) / DT
                avg_rates.append(rate)
                print(f'Injecting {name} at {rate} kg/day.')
            
            mass_components_n = mass_components
            
    stop = time.time()
    print("Runtime = %3.2f sec" % (stop - start))
    m.print_timers()

    nx, nz = m.reservoir.nx, m.reservoir.nz  # grid dimensions
    prop_list = m.physics.vars + list(m.physics.property_containers[0].output_props.keys())
    # M_H2O = m.physics.property_containers[0].Mw[0] # molar mass water in kg/kmol
    # M_CO2 = m.physics.property_containers[0].Mw[1] # molar mass CO2 in kg/kmol
    # PV = np.array(m.reservoir.mesh.volume)[1] * np.array(m.reservoir.mesh.poro) # pore volume
    # PV = PV[:m.reservoir.n]
    # Generate some example data to animate
    time_vector, property_array = m.output_properties(output_properties = prop_list, timestep = None)
    
    mass_data = []
    for i in range(len(time_vector)):
        data_array = [property_array[key][i] for key in prop_list]
        mass_per_component = m.get_mass_components(data_array)
        mass_co2 = mass_per_component['CO2']/1e6
        
        # # vapour mass fraction
        # wco2 = property_array['yCO2'][i] * M_CO2 / (property_array['yCO2'][i] * M_CO2 + (1 - property_array['yCO2'][i]) * M_H2O)
    
        # # mass of CO2 in the aqueous phase
        # mass_co2_aq = PV * (1 - property_array['satV'][i]) * property_array['xCO2'][i] * property_array['rho_mA'][i] * M_CO2
    
        # # mass of CO2 in the vapor phase
        # mass_co2_v = PV * property_array['satV'][i] * property_array['rhoV'][i] * wco2
    
        # # total mass of CO2 in kton
        # mass_co2 = (mass_co2_aq + mass_co2_v)/1e6
    
        # append to data list for plotting
        mass_data.append(mass_co2.reshape(m.reservoir.nz, m.reservoir.nx))
        
    plt.figure(dpi = 100, figsize = (10, 2))
    plt.title(f'Mass of $CO_2$ in kt @ year {time_vector[-1]/365}')
    c = plt.pcolor(grid[0], grid[1], mass_data[-1], cmap = 'jet')
    plt.colorbar(c, aspect = 10)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    # plt.tight_layout()
    plt.savefig(os.path.join(m.output_folder, f'year_{int(time_vector[-1]/365)}_mass_co2.png'), bbox_inches='tight')
    plt.close()
        
    solution_vector = np.array(m.physics.engine.X)
    for i, name in enumerate(prop_list):
        plt.figure(dpi = 100, figsize = (10, 2))
        plt.title(name + f' @ year {int(time_vector[-1]/365)}')
        c = plt.pcolor(grid[0], grid[1], data_array[i][:m.reservoir.n].reshape(m.reservoir.nz, m.reservoir.nx), cmap = 'jet')
        plt.colorbar(c, aspect = 10)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.savefig(os.path.join(m.output_folder, f'year_{int(time_vector[-1]/365)}_{name}.png'), bbox_inches='tight')
        plt.close()

    # process measured rates
    avg_rates = np.array(avg_rates)
    for i, name in enumerate(specs['components']):
        avg_rates[i::m.nc][25:50] /= 2 
    np.save(os.path.join(specs['output_dir'], 'avg_rates.npy'), avg_rates)
    
    # Define target rates
    target_rates = np.ones(len(avg_rates))
    target = avg_rates[:m.nc]
    for i, name in enumerate(specs['components']):
        target_rates[i::m.nc][:25] *= target[i]
        target_rates[i::m.nc][25:50] *= target[i]
        target_rates[i::m.nc][50:] = 0

    # total_rate = np.sum(np.array([avg_rates[i::m.nc] for i in range(m.nc-1)]), axis = 0) # sum of masses 
    
    # Compute percentage deviation, avoiding division by zero
    # percentage_deviation = np.where(target_rates != 0, ((avg_rates - target_rates) / target_rates) * 100, np.nan)
    
    for i, name in enumerate(specs['components']):
        
        fig, ax1 = plt.subplots(dpi=100)
        ax1.grid()
        ax2 = ax1.twinx()
        ax1.step(time_vector/365, np.hstack([avg_rates[i::m.nc], avg_rates[i::m.nc][-1]]), '-xb', where = 'post', label=f"Avg Rates {name}")
        # ax2.step(time_vector[1:]/365, percentage_deviation[i::m.nc], '--r', label=f"Percentage Deviation (%) {name}")
        # ax1.axhline(y=3024, color='k', linestyle='--', label='3024 m³/day')
        # ax1.axhline(y=2*3024, color='k', linestyle='--', label='2 * 3024 m³/day')
        ax1.set_xlim(min(time_vector/365), max(time_vector/365))
        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel(f"{name} - Injection Rate (kg/day)")#, color='b')
        ax1.tick_params(axis='y')#, labelcolor='b')
        ax2.set_ylabel("Deviation (%)")#, color='r')
        ax2.tick_params(axis='y')#, labelcolor='r')
        # ax1.legend()
        # ax2.legend()
        plt.title("Injection Rates & Percentage Deviation")
        plt.savefig(os.path.join(m.output_folder, f'{name}_rates.png'), bbox_inches='tight')
        plt.close()