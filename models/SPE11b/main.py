"""
This CICD model tests the SPE11b model with isothermal physics.

Additionally, it tests the post-processing, regions and restart capabilities of DARTS.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import platform as py_platform
import sys

# from model_b import Model, PorPerm, Corey, layer_props
from model_b import Model
from darts.engines import redirect_darts_output, sim_params
from darts.engines import well_control_iface

try:
    from darts.engines import set_gpu_device
except ImportError:
    pass
from fluidflower_str_b import FluidFlowerStruct

#%%

def output(m, ts, property_data : int = None):
    # save reservoir solution
    # m.output.save_data_to_h5('reservoir')

    # evaluate base properties
    if property_data is None:
        time_vector, property_array = m.output.output_properties(
            output_properties=m.physics.vars + m.output.properties,
            # ts_idx=-1
            engine = True
        )
    else:
        time_vector, property_array = property_data[0], property_data[1]
    # m.output.output_to_vtk(ith_step = ts, output_data = [time_vector, property_array])

    last_comp = m.physics.components[-1]
    property_array[last_comp] = 1.
    for name in m.physics.components[:-1]:
        property_array[last_comp] -= property_array[name]
    
    # compute mass per component
    mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
    nb = m.reservoir.mesh.n_res_blocks
    property_array['mass_CO2'] = mass_per_component['CO2'].reshape(1, nb) / 1e6
    property_array['mass_aqueous_CO2'] = mass_aqueous['CO2'].reshape(1, nb) / 1e6
    property_array['mass_vapor_CO2'] = mass_vapor['CO2'].reshape(1, nb) / 1e6

    # add units to unit dictionary for plotting purposes
    m.output.variable_units['mass_CO2'] = 'kt'
    m.output.variable_units['mass_aqueous_CO2'] = 'kt'
    m.output.variable_units['mass_vapor_CO2'] = 'kt'

    if m.specs['dispersion']:
        # store and plot phase velocities
        darcy_velocities = np.asarray(m.physics.engine.darcy_velocities).reshape(nb, m.physics.nph, 3)  # cell centered velocities
        for p, ph in enumerate(m.physics.phases): # per phase
            for v, orientation in enumerate(['x', 'y', 'z']): # per direction
                property_array[f'vel_{ph}_{orientation}'] = darcy_velocities[:, p, v].reshape(1, nz * nx)

            property_array[f'vel_{ph}'] = np.sqrt(np.square(property_array[f'vel_{ph}_x']) \
                                                    + np.square(property_array[f'vel_{ph}_y']) \
                                                        + np.square(property_array[f'vel_{ph}_z']))

        if m.specs['dispersion'] and m.specs['platform'] == 'cpu' and m.specs['RHS'] is True: # flux output is only enabled for CPU platform
            # store and plot fluxes
            diff = np.asarray(m.physics.engine.diffusion_fluxes) # array containing diffusive fluxes per component and phase
            disp = np.asarray(m.physics.engine.dispersion_fluxes) # array containing dispersion fluxes per component and phase
            darcy = np.asarray(m.physics.engine.darcy_fluxes) # array containing darcy fluxes per component and phase

            mult = m.physics.nc * m.physics.nph
            for id_key in ['SN']:
                for phase_idx, phase_name in enumerate(m.physics.phases):
                    for comp_idx, comp_name in enumerate(m.components):
                        i = phase_idx * m.physics.nc + comp_idx
                        property_array[f'diff_fluxes_{phase_name}_{comp_name}_{id_key}'] = diff[
                            mult * m.ids_list[id_key] + i].reshape(-1, nb)
                        property_array[f'darcy_fluxes_{phase_name}_{comp_name}_{id_key}'] = darcy[
                            mult * m.ids_list[id_key] + i].reshape(-1, nb)
                        property_array[f'disp_fluxes_{phase_name}_{comp_name}_{id_key}'] = disp[
                            mult * m.ids_list[id_key] + i].reshape(-1, nb)

            # m.plot_fluxes(property_array, time_vector, ts)  # plot fluxes
    if m.specs['ny'] == 1:
        m.plot_properties(property_array, time_vector, ts)  # plot properties
    else: 
        pass
    
    m.output.output_to_vtk(ith_step = ts, output_data = [time_vector, property_array])

    # save properties to HDF5 file
    if 1:
        # append data from all timesteps to a reservoir_solution.h5
        m.output.save_property_array(time_vector, property_array)
    else:
        # save data at ts in a specific h5 file
        m.output.save_property_array(time_vector, property_array, f'property_array_ts{ts}.h5')

def post_process(m, specs):

    vtk_array = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 36, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    )  # years for which to export a .vtk file
    output_props = m.physics.vars + m.output.properties

    time_vector, property_array = m.output.output_properties(
        sol_filepath=os.path.join(specs['output_dir'], 'reservoir_solution.h5'),
        output_properties=output_props,
        ts_idx=-1
    )

    avg_rates = []
    mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
    mass_components_n = {key: np.sum(value) for key, value in
                         mass_per_component.items()}  # sum over grid blocks per component

    if 0:
        for ts, year in enumerate(vtk_array):
            try:
                time_vector, property_array = m.output.output_properties(
                    sol_filepath = os.path.join(specs['output_dir'], 'reservoir_solution.h5'),
                    output_properties = props,
                    ts_idx = ts
                )
                m.plot_properties(property_array, time_vector, year)
                m.output.output_to_vtk(ith_step = year, output_data=[time_vector, property_array])
                print(f'Processed saved timestep {ts} corresponding to day {time_vector[0]}')

            except:
                print(f'No solution for year {year}')

        # # if you do this it is quicker as all the data will be output to .vtk at once but the numbers in the solution_ts{year}.vts will not match the year
        # # instead check the pvd file for the corresponding timestamp.
        # time_vector, property_array = m.output.output_properties(
        #     sol_filepath=os.path.join(specs['output_dir'], 'reservoir_solution.h5'),
        # )
        # m.output.output_to_vtk(sol_filepath=os.path.join(specs['output_dir'], 'reservoir_solution.h5'))

    if 1:
        restart_data_file_path = os.path.join(specs['output_dir'], 'reservoir_solution.h5')
        m.load_restart_data(reservoir_filepath = restart_data_file_path, ts_idx=-1)
        m.output.verbose = False

        event1 = True
        event2 = True

        if m.physics.engine.t < 25 * Dt:
            m.inj_rate = [specs['inj_rate'], 0]
        elif m.physics.engine.t >= 25 * Dt and m.physics.engine.t < 50 * Dt:
            m.inj_rate = [specs['inj_rate'], specs['inj_rate']]
            event1 = False
        elif m.physics.engine.t >= 50 * Dt and event2:
            m.inj_rate = [0, 0]
            event2 = False

        if specs['RHS']:
            event1, event2 = m.set_well_rhs(Dt, specs['inj_rate'], event1, event2)
        else:
            event1, event2 = m.set_well_rates(Dt, specs['inj_rate'], event1, event2)

        print(f'<<<<<<<<< Starting simulation at {m.physics.engine.t/365} with {m.inj_rate} >>>>>>>>>>>')
        start_ts = int(m.physics.engine.t//Dt)
        for ts in range(start_ts, Nt + 1):

            print(f'------------------- Simulate from year {(ts*Dt)/365} until year {((ts+1)*Dt)/365} ----------------------')
            m.run(Dt,
                  restart_dt=1.0,
                  save_reservoir_data=False,
                  save_well_data= not m.specs['RHS'],
                  save_well_data_after_run= not m.specs['RHS'],
                  verbose=True
                  )

            if specs['check_rates']:
                time_vector, property_array = m.output.output_properties(output_properties = m.physics.vars + m.output.properties, engine=True)
                mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
                mass_components = {key: np.sum(value) for key, value in mass_per_component.items()}

                for i, name in enumerate(m.components):
                    rate = (mass_components[name] - mass_components_n[name]) / Dt
                    avg_rates.append(rate)
                    print(f'Injecting {name} at {avg_rates[i::m.nc][-1]} kg/day.')
                mass_components_n = mass_components

            if m.physics.engine.t/Dt in vtk_array:
                output(m, ts + 1)

            if specs['RHS']:
                event1, event2 = m.set_well_rhs(Dt, specs['inj_rate'], event1, event2)
            else:
                event1, event2 = m.set_well_rates(Dt, specs['inj_rate'], event1, event2)

    return

def pkl_suffix():
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
        return '_gpu'
    elif os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
        return '_iter'
    else:
        return '_odls'

def run(m, specs, OUTPUT = 0):
    if specs['ny'] == 1:
        m.plot_reservoir()
        
    output_props = m.physics.vars + m.output.properties
    time_vector, property_array = m.output.output_properties(output_properties=output_props, ts_idx=0)
    m.output.output_to_vtk(ith_step = 0, output_data = [time_vector, property_array])

    avg_rates = []
    if specs['check_rates']:
        # time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=0)
        # m.output.save_property_array(time_vector, property_array, 'property_array_ts0.h5')
        m.output.append_properties_to_reservoir(time_vector, property_array)
        mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
        mass_components_n = {key: np.sum(value) for key, value in mass_per_component.items()} # sum over grid blocks per component

    # years for which to export a .vtk file
    vtk_array = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    )

    event1 = True
    event2 = True

    for ts in range(Nt):

        print(f'----------------------------------- Simulate from year {(ts*Dt)/365} until year {((ts+1)*Dt)/365} -----------------------------------')
        m.run(Dt,
              # restart_dt = Dt/10,
              save_reservoir_data = False,
              save_well_data = not m.specs['RHS'],
              save_well_data_after_run = not m.specs['RHS'],
              verbose = True
              )

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
                m.output.save_data_to_h5('reservoir')
                if OUTPUT:
                    output(m, ts + 1, property_data = [time_vector, property_array])

        else:
            if m.physics.engine.t/Dt in vtk_array:
                m.output.save_data_to_h5('reservoir')
                if OUTPUT:
                    output(m, ts + 1)

        if specs['RHS']:
            event1, event2 = m.set_well_rhs(Dt, specs['inj_rate'], event1, event2)
        else:
            event1, event2 = m.set_well_rates(Dt, specs['inj_rate'], event1, event2)

    return avg_rates

#%%

"""Define realization ID"""
Nt = 1
Dt = 25 * 365
nx = 840 // 5
nz = 120 // 5
zero = 1e-10

if 0:
    # --- model_specs wiring (put near imports) ---
    import os, json, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--model-specs', help='JSON string for model specs')
    parser.add_argument('--model-specs-file', help='Path to JSON file with model specs')
    args = parser.parse_args()

    def _load_model_specs():
        if args.model_specs:
            return json.loads(args.model_specs)
        if args.model_specs_file:
            with open(args.model_specs_file) as f:
                return json.load(f)
        if 'MODEL_SPECS' in os.environ:
            return json.loads(os.environ['MODEL_SPECS'])
        raise RuntimeError(
            "model_specs not provided. Use --model-specs '<JSON>' or "
            "--model-specs-file path.json or set $MODEL_SPECS."
        )

    model_specs = _load_model_specs()
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown_env")

    for spec in model_specs:
        base_out = spec.get("output_dir", "OUTPUT")
        spec["output_dir"] = os.path.join(base_out, env_name)
        #spec["inj_stream"] = [1.0 - zero, zero, 283.15]
        spec["inj_stream"] = [zero, 1-zero, 283.15]

else:
    # cpu/gpu based on platform
    platform = 'cpu' # default is CPU.
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
       platform = 'gpu'

    # please read the README file for an explanation of the input parameters :
    model_specs = [
        # SPE11b 
        {'check_rates': False, 'temperature': None, '1000years': 10, 'RHS': True,
             'components': ['H2O', 'CO2'], 'inj_stream': [0.001, .999, 283.15], 'inj_rate': 3024,
                 'nx': nx, 'nz': nz, 'ny': 1, 'dispersion': False, 'output_dir': 'new_output',
                     'post_process': None, 'platform': platform,
                         'parallel_evaluation': True, 'n_workers': 4
                         },
        
        # {'check_rates': True, 'temperature': None, '1000years': False, 'RHS': True,
        #     'components': ['H2O', 'CO2'], 'inj_stream': [0., 1., 283.15], 'inj_rate': 50 * 24 * 60 * 60,
        #         'nx': 50, 'nz': 50, 'ny': 10, 'dispersion': False, 'output_dir': 'SPE11c_results',
        #             'post_process': None, 'platform': 'cpu'},
        
        # restart model
        # {'check_rates': True, 'temperature': None, '1000years': None, 'RHS': True,
        #         'components': ['H2O', 'CO2'], 'inj_stream': [0., 1., 283.15], 'inj_rate': 3024, 
        #             'nx': nx, 'nz': nz, 'ny': 1, 'dispersion': False, 'output_dir': 'OUTPUT',
        #                 'post_process': 'POST', 'platform': 'cpu'},
    ]

if __name__ == '__main__':
    failed = 0
    for specs in model_specs:

        """ set up output directory """
        from model_b import build_output_dir
        if specs['output_dir'] is None:
            specs["output_dir"] = build_output_dir(specs)
        else:
            specs["output_dir"] = os.path.join(specs['output_dir'], build_output_dir(specs))

        if specs['post_process'] is None:
            output_dir = specs['output_dir']
        else:
            output_dir = os.path.join(specs['output_dir'], specs['post_process'])

        # save specs to a .txt file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'specs.pkl'), 'wb') as f:
            pickle.dump(specs, f)

        if 1:
            redirect_darts_output(os.path.join(output_dir, 'model.log'))
        else:
            from darts.tools.logging import redirect_all_output
            log_stream = redirect_all_output(os.path.join(output_dir, 'model.log'))

        m = Model(specs) 
        m.output_dir = output_dir
        m.print_darts()

        if specs['post_process'] is None:
            # ---- RUN MODEL
            m.init(
                discr_type='tpfa',
                platform=m.platform,
                verbose=True,
                parallel_evaluation=specs.get('parallel_evaluation', False),
                n_workers=specs.get('n_workers'),
            )
            
            # if specs['reservoir_type'] == '11c':
                # m.set_boundary_conditions_11c()
            
            m.print_stat()
            m.set_output(output_folder = m.output_dir, sol_filename = 'reservoir_solution.h5',
                         save_initial = not specs['1000years'], precision = 'd', verbose = False)
            # m.output.set_phase_properties()
            m.output.set_units() # adds unit labels to the vtk files 
            # m.output.print_simulation_parameters()
            
            if specs['dispersion']:
                m.init_dispersion()
                if specs['platform'] == 'cpu' and specs['RHS'] is True:
                    m.physics.engine.enable_flux_output()
                    m.map_mesh_faces()

            # simulate a thousand years
            n_years = specs.get("1000years", 0)
            if n_years > 0:
                for i in range(1, n_years + 1):
                    print(f"-------- Year {i}/{n_years} --------")
                    m.run(365, restart_dt=365,
                          save_reservoir_data=False,
                          save_well_data_after_run = True, 
                          save_well_data=False
                          )
                m.physics.engine.t = 0.0
                m.output.save_data_to_h5(kind="reservoir")
                m.inj_rate = [specs['inj_rate'], 0]
                if m.specs['RHS'] is False: 
                    m.set_well_controls()
            # m.output.verbose = False

            avg_rates = run(m, specs)

            if specs['RHS'] is False:
                time_data = m.output.store_well_time_data(
                    phase_molar_rates = False,
                    phase_mass_rates = False,
                    phase_volumetric_rates = False,
                    component_molar_rates = False,
                    component_mass_rates = True,
                    advective_heat_rates = False,
                    save_output_files = True
                )
                m.output.plot_well_time_data()

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
            """ PROCESS RESULTS AND RESTART MODEL """

            print(f'Post processing into {m.output_dir}...')
            m.init(
                discr_type='tpfa',
                platform=m.platform,
                parallel_evaluation=specs.get('parallel_evaluation', False),
                n_workers=specs.get('n_workers'),
            )
            m.set_output(output_folder = m.output_dir, sol_filename = 'reservoir_solution_PART2.h5',
                         save_initial=False, precision='d', verbose = False)
            # m.output.set_phase_properties()
            # new_prop_keys = ['dens_Aq', 'dens_V', 'sat_V', 'densm_Aq', 'enthalpy_V']
            # for ph in m.physics.phases:
            #     for comp in m.physics.components:
            #         new_prop_keys.append(f'x_{ph}_{comp}')
            # m.output.filter_phase_props(new_prop_keys)
            m.output.set_units()
            m.output.print_simulation_parameters()

            if specs['dispersion']:
                m.init_dispersion()
                if specs['platform'] == 'cpu' and specs['RHS'] is True:
                    m.physics.engine.enable_flux_output()
                    m.map_mesh_faces()

            avg_rates = post_process(m, specs)

        m.print_timers()
        m.print_stat()

        suffix = pkl_suffix()
        perf_file = os.path.join('ref', 'perf_' + py_platform.system().lower()[:3] + suffix + '.pkl')
        overwrite = 0
        if os.getenv('UPLOAD_PKL') == '1':
            overwrite = 1
        failed += m.check_performance(perf_file=perf_file, overwrite=overwrite, pkl_suffix=suffix)

    sys.exit(failed)
