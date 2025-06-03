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

def output(m, ts):
    # save reservoir solution
    m.output.save_data_to_h5('reservoir')

    # evaluate base properties
    time_vector, property_array = m.output.output_properties(output_properties=m.physics.vars + m.output.properties,
                                                             timestep=-1)

    # compute mass per component
    mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
    property_array['mass_CO2'] = mass_per_component['CO2'].reshape(1, nx * nz) / 1e6
    property_array['mass_aqueous_CO2'] = mass_aqueous['CO2'].reshape(1, nx * nz) / 1e6
    property_array['mass_vapor_CO2'] = mass_vapor['CO2'].reshape(1, nx * nz) / 1e6

    # add units to unit dictionary for plotting purposes
    m.output.variable_units['mass_CO2'] = 'kt'
    m.output.variable_units['mass_aqueous_CO2'] = 'kt'
    m.output.variable_units['mass_vapor_CO2'] = 'kt'

    if m.specs['dispersion']:
        # store and plot phase velocities
        darcy_velocities = np.asarray(m.physics.engine.darcy_velocities).reshape(m.reservoir.mesh.n_res_blocks,
                                                                                 m.physics.nph,
                                                                                 3)  # cell centered velocities
        for p, ph in enumerate(m.physics.phases): # per phase
            for v, orientation in enumerate(['x', 'y', 'z']): # per direction
                property_array[f'vel_{ph}_{orientation}'] = darcy_velocities[:, p, v].reshape(1, nz * nx)
                # max_velocities.append(np.max(darcy_velocities[:, p, v]))

            property_array[f'vel_{ph}'] = np.sqrt(np.square(property_array[f'vel_{ph}_x']) \
                                                  + np.square(property_array[f'vel_{ph}_y']) \
                                                  + np.square(property_array[f'vel_{ph}_z']))

        # store and plot fluxes
        diff = np.asarray(m.physics.engine.diffusion_fluxes) # array containing diffusive fluxes per component and phase
        disp = np.asarray(m.physics.engine.dispersion_fluxes) # array containing dispersion fluxes per component and phase
        darcy = np.asarray(m.physics.engine.darcy_fluxes) # array containing darcy fluxes per component and phase

        mult = m.physics.nc * m.physics.nph
        for id_key in ['SN', 'EW']:
            for phase_idx, phase_name in enumerate(m.physics.phases):
                for comp_idx, comp_name in enumerate(m.components):
                    i = phase_idx * m.physics.nc + comp_idx
                    property_array[f'diff_fluxes_{phase_name}_{comp_name}_{id_key}'] = diff[
                        mult * m.ids_list[id_key] + i]
                    property_array[f'darcy_fluxes_{phase_name}_{comp_name}_{id_key}'] = darcy[
                        mult * m.ids_list[id_key] + i]
                    property_array[f'disp_fluxes_{phase_name}_{comp_name}_{id_key}'] = disp[
                        mult * m.ids_list[id_key] + i]

        # m.plot_fluxes(property_array, time_vector, ts)
        # m.plot_properties(property_array, time_vector, ts)
        # m.my_output_to_vtk(property_array=property_array, timesteps=time_vector, ith_step=ts)

    m.plot_fluxes(property_array, time_vector, ts)  # plot fluxes
    m.plot_properties(property_array, time_vector, ts)  # plot properties
    # m.my_output_to_vtk(property_array=property_array, timesteps=time_vector, ith_step=ts)  # output to vtk
    m.output.save_property_array(time_vector, property_array, f'property_array_ts{ts}.h5')

def post_process(m, specs):

    vtk_array = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 36, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    )  # years for which to export a .vtk file
    props = m.physics.vars + m.output.properties

    for ts, year in enumerate(vtk_array):
        try:
            time_vector, property_array = m.output.output_properties(
                filepath = os.path.join(specs['output_dir'], 'reservoir_solution.h5'),
                output_properties = props,
                timestep = ts
            )
            m.plot_properties(property_array, time_vector, year)
            print(f'Process {ts} at year {year}')

        except:
            print(f'No solution for year {year}')

    if 1:
        restart_data_file_path = os.path.join(specs['output_dir'], 'reservoir_solution.h5')
        m.load_restart_data(reservoir_filename = restart_data_file_path, timestep = -1)

        # m.output.set_phase_properties()
        # new_prop_keys = ['dens_Aq', 'dens_V', 'enthalpy_V', 'sat_V', 'densm_Aq']
        # for ph in m.physics.phases:
        #     for comp in m.physics.components:
        #         new_prop_keys.append(f'x_{ph}_{comp}')
        # m.output.filter_phase_props(new_prop_keys)
        # m.output.set_units()

        m.output.verbose = False

        event1 = True
        event2 = True

        if m.physics.engine.t >= 25 * Dt and m.physics.engine.t < 50 * Dt:
            m.inj_rate = [3024, 3024]
            event1 = False
        elif m.physics.engine.t >= 50 * Dt and event2:
            m.inj_rate = [0, 0]
            event2 = False

        if specs['RHS']:
            event1, event2 = m.set_well_rhs(Dt, 3024, event1, event2)
        else:
            event1, event2 = m.set_well_rates(Dt, 3024, event1, event2)

        print(f'<<<<<<<<< Starting simulation at {m.physics.engine.t} with {m.inj_rate} >>>>>>>>>>>')
        start_ts = int(m.physics.engine.t//Dt)
        for ts in range(start_ts, Nt):
            print(f'------------------- Simulate from year {(ts*Dt)/365} until year {((ts+1)*Dt)/365} ----------------------')
            m.run(Dt, restart_dt = 1.0, save_reservoir_data = False, save_well_data = not specs['RHS'], verbose=True)

            if m.physics.engine.t / Dt in vtk_array:
                output(m, ts + 1)

    return

def run(m, specs):
    m.plot_reservoir()

    output_props = m.physics.vars + m.output.properties
    m.output.output_to_vtk(output_properties=output_props, ith_step=0)

    avg_rates = []
    max_velocities = []
    if specs['check_rates']:
        time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=0)
        m.output.save_property_array(time_vector, property_array, 'property_array_ts0.h5')
        mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
        mass_components_n = {key: np.sum(value) for key, value in mass_per_component.items()} # sum over grid blocks per component

    # years for which to export a .vtk file
    vtk_array = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    event1 = True
    event2 = True

    ids_list, ids_cells_list = m.map_mesh_faces()

    for ts in range(Nt):
        # print(f'----------------------------------- Simulate from year {(ts*Dt)/365} until year {((ts+1)*Dt)/365} -----------------------------------')
        start_year = (ts * Dt) / 365
        end_year = ((ts + 1) * Dt) / 365

        print(f'''
       
           Simulating from year {start_year:.4f} to year {end_year:.4f}  
        ╰───────────────────────────────────────────────────────────────╯
        ''')

        m.run(Dt, restart_dt = 1.0, save_reservoir_data = False, save_well_data = not specs['RHS'], verbose=True)


        if specs['check_rates']:
            time_vector, property_array = m.output.output_properties(output_properties = output_props, engine=True)
            mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
            mass_components = {key: np.sum(value) for key, value in mass_per_component.items()}

            for i, name in enumerate(m.components):
                rate = (mass_components[name] - mass_components_n[name]) / Dt
                avg_rates.append(rate)
                print(f'Injecting {name} at {avg_rates[i::m.nc][-1]} kg/day.')
            mass_components_n = mass_components

        if m.physics.engine.t/Dt in vtk_array:
            output(m, ts + 1)
        #     m.output.save_data_to_h5('reservoir') # save reservoir state
        #     time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=-1) # evaluate base properties
        #
        #     # compute mass per component
        #     mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
        #     property_array['mass_CO2'] = mass_per_component['CO2'].reshape(1, nx * nz) / 1e6
        #     property_array['mass_aqueous_CO2'] = mass_aqueous['CO2'].reshape(1, nx * nz) / 1e6
        #     property_array['mass_vapor_CO2'] = mass_vapor['CO2'].reshape(1, nx * nz) / 1e6
        #
        #     # add units to unit dictionary for plotting purposes
        #     m.output.variable_units['mass_CO2'] = 'kt'
        #     m.output.variable_units['mass_aqueous_CO2'] = 'kt'
        #     m.output.variable_units['mass_vapor_CO2'] = 'kt'
        #
        #     if specs['dispersion']:
        #         property_array_filepath = os.path.join(m.output_folder, 'property_arrays')
        #         os.makedirs(property_array_filepath, exist_ok=True)
        #
        #         # store and plot phase velocities
        #         darcy_velocities = np.asarray(m.physics.engine.darcy_velocities).reshape(m.reservoir.mesh.n_res_blocks, m.physics.nph, 3) # cell centered velocities
        #         for p, ph in enumerate(m.physics.phases): # per phase
        #             for v, orientation in enumerate(['x', 'y', 'z']): # per direction
        #                 property_array[f'vel_{ph}_{orientation}'] = darcy_velocities[:, p, v].reshape(1, nz * nx)
        #                 max_velocities.append(np.max(darcy_velocities[:, p, v]))
        #
        #             property_array[f'vel_{ph}'] = np.sqrt(np.square(property_array[f'vel_{ph}_x'])\
        #                                                   + np.square(property_array[f'vel_{ph}_y'])\
        #                                                     + np.square(property_array[f'vel_{ph}_z']))
        #
        #         # store and plot fluxes
        #         diff = np.asarray(m.physics.engine.diffusion_fluxes) # array containing diffusive fluxes per component and phase
        #         disp = np.asarray(m.physics.engine.dispersion_fluxes) # array containing dispersion fluxes per component and phase
        #         darcy = np.asarray(m.physics.engine.darcy_fluxes) # array containing darcy fluxes per component and phase
        #
        #         mult = m.physics.nc * m.physics.nph
        #         for id_key in ['SN', 'EW']:
        #             for phase_idx, phase_name in enumerate(m.physics.phases):
        #                 for comp_idx, comp_name in enumerate(m.components):
        #                     i = phase_idx * m.physics.nc + comp_idx
        #                     property_array[f'diff_fluxes_{phase_name}_{comp_name}_{id_key}'] =  diff[mult * m.ids_list[id_key] + i]
        #                     property_array[f'darcy_fluxes_{phase_name}_{comp_name}_{id_key}'] = darcy[mult * m.ids_list[id_key] + i]
        #                     property_array[f'disp_fluxes_{phase_name}_{comp_name}_{id_key}'] = disp[mult * m.ids_list[id_key] + i]
        #
        #         m.plot_fluxes(property_array, time_vector, ts + 1)
        #         m.plot_properties(ts + 1, time_vector, property_array)
        #         m.my_output_to_vtk(property_array=property_array, timesteps=time_vector, ith_step=ts + 1)
        #
        #     m.plot_fluxes(property_array, time_vector, ts + 1) # plot fluxes
        #     m.plot_properties(ts + 1, time_vector, property_array) # plot properties
        #     m.my_output_to_vtk(property_array=property_array, timesteps=time_vector, ith_step=ts + 1) # output to vtk
        #
        #     # save property array
        #     m.output.save_property_array(time_vector, property_array, os.path.join(property_array_filepath, f'property_array_ts{ts + 1}.h5'))

        if specs['RHS']:
            event1, event2 = m.set_well_rhs(Dt, 3024, event1, event2)
        else:
            event1, event2 = m.set_well_rates(Dt, 3024, event1, event2)

    return avg_rates

#%%

"""Define realization ID"""
Nt = 1000
Dt = 365
nx = 840//4
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
         
        # RHS CORRECTION WITH DISPERSION ON
    {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
        'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': 'OUTPUT', 'post_process': 'POST', 'gpu_device': False},

    #     # WELLS WITH DISPERSION OFF
    # {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'], 'inj_stream': [1-zero, 283.15],
    #     'nx': nx, 'nz': nz, 'dispersion': False, 'output_dir': None, 'post_process': None, 'gpu_device': False},
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

        m = Model(specs) # instantiate DartsModel
        m.print_darts()

        if specs['post_process'] is None:

            """ RUN MODEL """

            m.init(discr_type='tpfa', platform=m.platform)
            m.set_output(output_folder = m.output_dir, sol_filename = 'reservoir_solution.h5',
                         save_initial = not specs['1000years'], precision = 'd', verbose = True)
            # m.output.set_phase_properties()
            # m.output.set_units()
            # m.output.print_simulation_parameters()

            if specs['dispersion']:
                m.init_dispersion()
            m.physics.engine.enable_flux_output()
            ids_list, ids_cells_list = m.map_mesh_faces()

            # simulate a thousand years
            if specs['1000years']:
                for i in range(100):
                    print(f'-------- Year {i}/100 --------')
                    m.run(365, restart_dt=365 / 2, save_reservoir_data=False, save_well_data=False)
                m.physics.engine.t = 0.
                m.output.save_data_to_h5(kind='reservoir')
                m.inj_rate = [3024, 0]
            m.output.verbose = False

            avg_rates = run(m, specs)
            m.print_timers()
            m.print_stat()
            # m.output.output_to_vtk(output_properties = m.physics.vars + m.output.properties)

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

        else:

            """ PROCESS SOME OF THE RESULTS """

            print(f'Post processing into {m.output_dir}...')
            m.init(discr_type = 'tpfa', platform = m.platform)
            m.set_output(output_folder = m.output_dir, sol_filename = 'reservoir_solution_PART2.h5',
                         save_initial=False, verbose = False)

            m.output.set_phase_properties()
            new_prop_keys = ['dens_Aq', 'dens_V', 'enthalpy_V', 'sat_V', 'densm_Aq']
            for ph in m.physics.phases:
                for comp in m.physics.components:
                    new_prop_keys.append(f'x_{ph}_{comp}')
            m.output.filter_phase_props(new_prop_keys)

            m.output.set_units()
            m.output.print_simulation_parameters()

            if specs['dispersion']:
                m.init_dispersion()

            m.physics.engine.enable_flux_output()
            ids_list, ids_cells_list = m.map_mesh_faces()

            avg_rates = post_process(m, specs)



