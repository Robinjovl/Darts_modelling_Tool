"""
This CICD model tests the SPE11b model with isothermal physics.

Additionally, it tests the post-processing, regions and restart capabilities of DARTS.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from model_b import Model, PorPerm, Corey, layer_props
from darts.engines import redirect_darts_output, sim_params, well_control_iface
from fluidflower_str_b import FluidFlowerStruct
from darts.models.cicd_model import compare_solution_with_reference, get_platform


#%%
def output(m, ts, property_data : int = None):
    # save reservoir solution
    m.output.save_data_to_h5('reservoir')

    # evaluate base properties
    if property_data is None:
        time_vector, property_array = m.output.output_properties(
            output_properties=m.physics.vars + m.output.properties,
            # timestep=-1
            engine = True
        )
    else:
        time_vector, property_array = property_data[0], property_data[1]
    m.output.output_to_vtk(output_data = [time_vector, property_array])

    # compute mass per component
    mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
    property_array['mass_CO2'] = mass_per_component['CO2'].reshape(1, nx * nz) / 1e6
    property_array['mass_aqueous_CO2'] = mass_aqueous['CO2'].reshape(1, nx * nz) / 1e6
    property_array['mass_vapor_CO2'] = mass_vapor['CO2'].reshape(1, nx * nz) / 1e6

    # add units to unit dictionary for plotting purposes
    m.output.variable_units['mass_CO2'] = 'kt'
    m.output.variable_units['mass_aqueous_CO2'] = 'kt'
    m.output.variable_units['mass_vapor_CO2'] = 'kt'

    if 1:
        # store and plot phase velocities
        darcy_velocities = np.asarray(m.physics.engine.darcy_velocities).reshape(m.reservoir.mesh.n_res_blocks,
                                                                                 m.physics.nph,
                                                                                 3)  # cell centered velocities
        for p, ph in enumerate(m.physics.phases): # per phase
            for v, orientation in enumerate(['x', 'y', 'z']): # per direction
                property_array[f'vel_{ph}_{orientation}'] = darcy_velocities[:, p, v].reshape(1, nz * nx)

            property_array[f'vel_{ph}'] = np.sqrt(np.square(property_array[f'vel_{ph}_x']) \
                                                    + np.square(property_array[f'vel_{ph}_y']) \
                                                        + np.square(property_array[f'vel_{ph}_z']))

        if 0: # flux output is only enabled for CPU platform
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
                            mult * m.ids_list[id_key] + i]
                        property_array[f'darcy_fluxes_{phase_name}_{comp_name}_{id_key}'] = darcy[
                            mult * m.ids_list[id_key] + i]
                        property_array[f'disp_fluxes_{phase_name}_{comp_name}_{id_key}'] = disp[
                            mult * m.ids_list[id_key] + i]

    if 0:
        m.plot_fluxes(property_array, time_vector, ts)  # plot fluxes
    m.plot_properties(property_array, time_vector, ts)  # plot properties

    # export properties
    if 1:
        # append data from all timesteps to a single file h5
        m.output.save_property_array(time_vector, property_array)
    else:
        # save data at ts in a specific h5 file
        m.output.save_property_array(time_vector, property_array, f'property_array_ts{ts}.h5')

def post_process(m, specs):

    vtk_array = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 36, 40, 45, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    )  # years for which to export a .vtk file
    props = m.physics.vars + m.output.properties
    output_props = m.physics.vars + m.output.properties

    time_vector, property_array = m.output.output_properties(
        filepath=os.path.join(specs['output_dir'], 'reservoir_solution.h5'),
        output_properties=output_props,
        timestep=-1
    )

    avg_rates = []
    mass_per_component, mass_vapor, mass_aqueous = m.get_mass_components(property_array)
    mass_components_n = {key: np.sum(value) for key, value in
                         mass_per_component.items()}  # sum over grid blocks per component

    if 0:
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
        m.output.verbose = False

        event1 = True
        event2 = True

        if m.physics.engine.t < 25 * Dt:
            m.inj_rate = [3024, 0]
        elif m.physics.engine.t >= 25 * Dt and m.physics.engine.t < 50 * Dt:
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
        for ts in range(start_ts, Nt + 1):

            print(f'------------------- Simulate from year {(ts*Dt)/365} until year {((ts+1)*Dt)/365} ----------------------')
            m.run(Dt, restart_dt = 1.0, save_reservoir_data = False, save_well_data = not specs['RHS'], verbose=True)

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
                event1, event2 = m.set_well_rhs(Dt, 3024, event1, event2)
            else:
                event1, event2 = m.set_well_rates(Dt, 3024, event1, event2)

    return

def run(m, specs, platform='cpu'):
    m.plot_reservoir()

    output_props = m.physics.vars + m.output.properties
    m.output.output_to_vtk(output_properties=output_props, ith_step=0)

    avg_rates = []
    if specs['check_rates']:
        time_vector, property_array = m.output.output_properties(output_properties=output_props, timestep=0)
        m.output.save_property_array(time_vector, property_array, 'property_array_ts0.h5')
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
                output(m, ts + 1, property_data = [time_vector, property_array])

        else:
            if m.physics.engine.t/Dt in vtk_array:
                output(m, ts + 1)

        if specs['RHS']:
            event1, event2 = m.set_well_rhs(Dt, 3024, event1, event2)
        else:
            event1, event2 = m.set_well_rates(Dt, 3024, event1, event2)

    return avg_rates

#%%

"""Define realization ID"""
Nt = 1
Dt = 365
nx = 840//10
nz = 120//10
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
    platform = 'cpu'
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
       platform = 'gpu'

    model_specs = [
            # RHS CORRECTION WITH DISPERSION ON
        {'check_rates': True, 'temperature': 273.15 + 40., '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'],
            'inj_stream': [0.99, 0.01, 283.15],
                'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': 'OUTPUT',
                    'post_process': None, 'platform': 'cpu'},

        {'check_rates': True, 'temperature': 273.15 + 40., '1000years': False, 'RHS': True, 'components': ['CO2', 'H2O'],
            'inj_stream': [0.99, 0.01, 283.15],
                'nx': nx, 'nz': nz, 'dispersion': True, 'output_dir': 'OUTPUT',
                    'post_process': 'POST', 'platform': 'cpu'},
    ]

if __name__ == '__main__':
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

        # save specs to a .pkl file
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
            """ RUN MODEL """
            m.init(discr_type='tpfa', platform=m.platform)
            m.print_stat()
            m.set_output(output_folder = m.output_dir, sol_filename = 'reservoir_solution.h5',
                         save_initial = not specs['1000years'], precision = 'd', verbose = True)
            # m.output.set_phase_properties()
            m.output.set_units()
            m.output.print_simulation_parameters()

            if specs['dispersion']:
                m.init_dispersion()

            if 0:
                m.physics.engine.enable_flux_output()
                m.map_mesh_faces()

            # simulate a thousand years
            if specs['1000years']:
                for i in range(specs['1000years']):
                    print(f'-------- Year {i}/100 --------')
                    m.run(365, restart_dt=365, save_reservoir_data=False, save_well_data=False)
                m.physics.engine.t = 0.
                m.output.save_data_to_h5(kind='reservoir')
                m.inj_rate = [3024, 0]
            # m.output.verbose = False

            avg_rates = run(m, specs)

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
            """ PROCESS RESULTS AND RESTART MODEL """

            print(f'Post processing into {m.output_dir}...')
            m.init(discr_type = 'tpfa', platform = m.platform)
            m.set_output(output_folder = m.output_dir, sol_filename = 'reservoir_solution_PART2.h5',
                         save_initial=False, precision='d', verbose = False)
            m.output.set_phase_properties()
            new_prop_keys = ['dens_Aq', 'dens_V', 'enthalpy_V', 'sat_V', 'densm_Aq', 'densm_V']
            for ph in m.physics.phases:
                for comp in m.physics.components:
                    new_prop_keys.append(f'x_{ph}_{comp}')
            m.output.filter_phase_props(new_prop_keys)
            m.output.set_units()
            m.output.print_simulation_parameters()

            if specs['dispersion']:
                m.init_dispersion()

            if specs['platform'] == 'cpu':
                m.physics.engine.enable_flux_output()
                m.map_mesh_faces()

            avg_rates = post_process(m, specs)

        m.print_timers()
        m.print_stat()


def run(platform='cpu'):
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

    redirect_darts_output('model.log') # redirects run.log to your directory of choice instead of prininting everything off

    """Define realization ID"""
    model_specs = [
        {'structured': True,
         'thickness': False,
         'curvature': False,
         'tpfa': True,
         'capillary': True,
         'nx': 170,
         'nz': 60,
         'output_dir': 'SPE11_output'},
    ]

    j = 0
    specs = model_specs[j]
    m = Model()
    m.platform = platform
    """Define physics"""
    zero = 1e-10
    m.set_physics(corey=corey, zero=zero, temperature=323.15, n_points=1001, diff=1e-9)

    # solver paramters
    m.set_sim_params(first_ts=1e-2, mult_ts=2, max_ts=365, tol_linear=1e-3, tol_newton=1e-3,
                     it_linear=50, it_newton=12, newton_type=sim_params.newton_global_chop)
    m.params.newton_params[0] = 0.05
    m.params.nonlinear_norm_type = m.params.L1

    """Define the reservoir and wells """
    well_centers = {
        "I1": [2700.0, 0.0, 300.0],
        "I2": [5100.0, 0.0, 700.0]
    }

    structured = specs['structured']
    m.reservoir = FluidFlowerStruct(timer=m.timer, layer_properties=layer_props, layers_to_regions=layers_to_regions,
                                    model_specs=specs, well_centers=well_centers) # structured reservoir

    if 0:
        grid = np.meshgrid(np.linspace(0, 8400, m.reservoir.nx), np.linspace(0, 1200, m.reservoir.nz))
        plt.figure(figsize = (10, 2))
        plt.title('Porosity')
        c = plt.pcolor(grid[0], grid[1], m.reservoir.global_data['poro'].reshape(m.reservoir.nz, m.reservoir.nx))
        plt.colorbar(c)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.show()

    """ Define initial and boundary conditions """
    # define initial pressure, composition and temperature of the reservoir
    pres_in = 212
    m.initial_values = {"pressure": pres_in,
                        "H2O": 1. - zero,
                        "temperature": 323.15}
    m.gradient = {"pressure": 0.09775, "temperature": 0.0}

    # define injection stream of the wells
    m.inj_stream = [zero]

    inj_rate = 3024 # mass rate per well, kg/day
    m.inj_rate = [0, 0] # per well

    m.set_str_boundary_volume_multiplier()  # right and left boundary volume multiplier

    # now that your reservoir and physics is defined, you can init your DartsModel()
    output_dir = specs['output_dir']

    m.init(discr_type='tpfa', platform=platform, output_folder=output_dir, restart = True)

    # equillibration step
    m.run_python_my(365)
    m.save_data_to_h5('solution')
    m.physics.engine.t = 0 # return engine time to zero

    m.inj_rate = [inj_rate, 0]  # [well 1, well 2]
    Nt = 20  # 100 years with output every 5 years
    import time

    start = time.time()
    for i in range(Nt):
        m.run_python_my(5*365)
        m.save_data_to_h5('solution')

        if m.physics.engine.t >= 25 * 365 and m.physics.engine.t < 50 * 365:
            # At 25 years, start injecting in the second well
            m.inj_rate = [inj_rate, inj_rate]

        elif m.physics.engine.t >= 50 * 365:
            # At 50 years, stop injection for both wells
            m.inj_rate = [0, 0]
    stop = time.time()
    print("Runtime = %3.2f sec" % (stop - start))

    nx, nz = m.reservoir.nx, m.reservoir.nz  # grid dimensions

    prop_list = list(m.physics.property_containers[0].output_props.keys())

    M_H2O = m.physics.property_containers[0].Mw[0] # molar mass water in kg/kmol
    M_CO2 = m.physics.property_containers[0].Mw[1] # molar mass CO2 in kg/kmol
    PV = np.array(m.reservoir.mesh.volume)[1] * np.array(m.reservoir.mesh.poro) # pore volume

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
        mass_co2 = (mass_co2_aq + mass_co2_v)/1e3

        # append to data list for plotting
        data.append(mass_co2.reshape(m.reservoir.nz, m.reservoir.nx))

    # from matplotlib.animation import FuncAnimation
    #
    # # Initialize the figure and color plot
    # fig, ax = plt.subplots(figsize=(16, 4))
    # pcolor_plot = ax.pcolor(grid[0], grid[1], data[0], vmin = data[1].min(), vmax = 25.0, cmap = 'cividis')
    # plt.colorbar(pcolor_plot, ax=ax)
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("z [m]")
    #
    # # Define the update function for animation
    # def update(frame):
    #     ax.set_title(f"mCO2 @ year {frame*5}")
    #     pcolor_plot.set_array(data[frame].ravel())  # Update the data
    #     return pcolor_plot,
    #
    # # Create the animation
    # ani = FuncAnimation(fig, update, frames=len(time_vector), blit=True)
    #
    # # Display the animation in the Jupyter Notebook
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())

    # for CI/CD
    failed, sim_time = compare_solution_with_reference(m=m)
    return failed

if __name__ == '__main__':
    exit(run(platform=get_platform()))
