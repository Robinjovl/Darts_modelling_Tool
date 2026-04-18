import os, signal, sys
os.environ["OMP_NUM_THREADS"] = "4"
import shutil
from model import Model
from darts.engines import redirect_darts_output
import numpy as np
from visualization import plot_profiles, plot_new_profiles, animate_1d

def run_simulation(domain: str, max_ts: float, nx: int = 100, mesh_filename: str = None, poro_filename: str = None,
                   output: bool = False, interpolator: str = 'multilinear', minerals: list = ['calcite'],
                   kinetic_mechanisms: list = ['acidic', 'neutral', 'carbonate'], output_folder: str = None,
                   n_obl_mult: int = 1, co2_injection: float = 0.1, h2o_injection: float = 1.1,
                   inj_rate: float = None, perm_poro: str = 'power_8', platform: str = 'cpu',
                   ni_dt_increase_cutoff: int = 5, ni_dt_decrease_cutoff: int = 8, n_good_ts: int = 10, report_timesteps = None,
                   flash: str = 'phreeqc', database: str = 'phreeqc'):
    # Make a folder
    if output_folder is None:
        output_folder = f'output_{domain}_{nx}_' + '_'.join(minerals) + \
            '_' + '_'.join(kinetic_mechanisms) + f'_{interpolator}_{n_obl_mult}'
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # Redirect output to log file
    redirect_darts_output(os.path.join(output_folder, 'log.txt'))

    # Create model
    m = Model(domain=domain, nx=nx, mesh_filename=mesh_filename, poro_filename=poro_filename,
              minerals=minerals, kinetic_mechanisms=kinetic_mechanisms, n_obl_mult=n_obl_mult,
              co2_injection=co2_injection, h2o_injection=h2o_injection, inj_rate=inj_rate,
              perm_poro=perm_poro, flash=flash, database=database)

    # Initialize model
    m.init(itor_type=interpolator, platform=platform, verbose=True, n_solid=len(minerals))
    m.set_output(output_folder=output_folder, sol_filename=f'nx{nx}.h5')

    # Initialization check
    if platform == 'cpu':
        op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
        # to calculate porosities for print
        poro_id = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name.index('porosity')
        m.physics.property_itor[0].evaluate_with_derivatives(m.physics.engine.X, m.physics.engine.region_cell_idx[0],
                                                                m.output.prop_values, m.output.prop_dvalues)
        poro = np.array([m.output.prop_values_np[poro_id::m.physics.n_property_itor_ops]])
        volume = np.array(m.reservoir.mesh.volume, copy=False)
        total_pv = np.sum(volume[:m.n_res_blocks] * poro) * 1e6
        print(f'Total pore volume: {total_pv} cm3')
        print(f'Injection rate: {m.inj_cells.size} cells * {m.inj_rate / total_pv * 1e+6} PV/day')

    # visualization
    def plot(m, ith_step=None):
        if output:
            if domain == '1D': return plot_profiles(m, output_folder=output_folder)
            else: m.output.output_to_vtk(ith_step=ith_step)
    # to report timers upon receiving SIGTERM signal
    def _term(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    try:
        m.n_good_ts = n_good_ts
        m.ni_dt_increase_cutoff = ni_dt_increase_cutoff
        m.ni_dt_decrease_cutoff = ni_dt_decrease_cutoff

        # intialization without injection
        if minerals == ['calcite']:
            init_days = 0.1
            num_time_iterations = 7
        elif set(minerals) == set(['calcite', 'dolomite']):
            init_days = 20.0
            num_time_iterations = 7
        else:
            init_days = 150.0
            num_time_iterations = 3

        rate = m.inj_rate
        m.inj_rate = 0.0
        m.data_ts.dt_max = 0.05
        m.run(days=init_days)

        # injection
        m.inj_rate = rate
        m.physics.engine.t = 0.0
        ith_step = 0
        m.data_ts.dt_max = max_ts
        if domain == '1D':
            m.data_ts.dt_first = 1.e-6
            m.data_ts.dt_mult = 1.5
            fig_paths = []
            fig_paths.append(plot(m))
            m.run(days=0.002, restart_dt=max_ts)
            fig_paths.append(plot(m))
            m.run(days=0.008, restart_dt=max_ts)
            fig_paths.append(plot(m))
            m.run(days=0.010, restart_dt=max_ts)
            fig_paths.append(plot(m))
            m.run(days=0.02, restart_dt=max_ts)
            fig_paths.append(plot(m))
            # m.data_ts.dt_max *= 3
            m.data_ts.first_ts = m.data_ts.dt_max
            m.run(days=0.1, restart_dt=max_ts)
            # m.data_ts.dt_max *= 4
            m.run(days=0.86)
            fig_paths.append(plot(m))
            # m.data_ts.dt_max *= 5
            m.data_ts.first_ts = m.data_ts.dt_max

            for i in range(num_time_iterations):
                dt = 2.0
                m.run(days=dt)
                if i < 1:
                    # m.data_ts.dt_max *= 1.5
                    m.data_ts.first_ts = m.data_ts.dt_max
                fig_paths.append(plot(m))

            # if output:
            #     animate_1d(output_folder=output_folder, fig_paths=fig_paths)
        elif domain in ('2D', '3D'):
            plot(m=m, ith_step=ith_step)
            ith_step += 1

            if report_timesteps is None:
                if domain == '2D':
                    report_timesteps = np.array([0.001, 0.001, 0.001, 0.002, 0.005,
                                                0.01, 0.01, 0.02, 0.05,
                                                0.1, 0.1, 0.2]) * 1e-4 / m.inj_rate
                    n_fine = 3
                    dt_max_bump = 1.0
                else:  # 3D
                    report_timesteps = np.array([0.001, 0.001, 0.001, 0.002, 0.005,
                                                0.01, 0.01, 0.02, 0.05,
                                                0.1, 0.1, 0.2]) * 0.0016128 / m.inj_rate
                    n_fine = 3
                    dt_max_bump = 30.0
                default_run = True
            else:
                report_timesteps = np.asarray(report_timesteps)
                n_fine = len(report_timesteps)
                dt_max_bump = 1.0
                default_run = False

            m.data_ts.dt_first = m.prev_dt = min(1.e-6 * 1e-3 / m.inj_rate, m.data_ts.dt_max)
            m.data_ts.dt_mult = 1.5
            ts_after_bt = 0
            max_snapshots_after_bt = 5
            for i, rts in enumerate(report_timesteps):
                if i == n_fine and dt_max_bump != 1.0:
                    m.data_ts.dt_max *= dt_max_bump
                    m.data_ts.first_ts = m.data_ts.dt_max
                m.run(days=rts, restart_dt=m.prev_dt)
                plot(m=m, ith_step=ith_step)
                ith_step += 1

                if default_run and m.reservoir.wh_propagation_ratio > 0.999:
                    ts_after_bt += 1
                    if ts_after_bt >= max_snapshots_after_bt:
                        break

            if default_run:
                while ts_after_bt < max_snapshots_after_bt:
                    m.run(days=report_timesteps.max(), restart_dt=m.prev_dt)
                    plot(m=m, ith_step=ith_step)
                    ith_step += 1

                    if m.reservoir.wh_propagation_ratio > 0.999:
                        ts_after_bt += 1

    finally:
        # Print some statistics
        print('\nNegative composition occurrence:', m.physics.reservoir_operators[0].counter, '\n')
        m.print_timers()
        m.print_stat()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # copy files to save configuration
        # shutil.copy('main.py', os.path.join(output_folder, 'main.py'))
        # shutil.copy('model.py', os.path.join(output_folder, 'model.py'))


def get_output_folder(args: dict):
    if not isinstance(args, dict):
        return 'run'
    if args.get('output_folder'):
        return str(args['output_folder'])
    parts = []
    for key in ['domain', 'nx', 'flash', 'database']:
        if key in args:
            parts.append(str(args[key]))
    return '_'.join(parts) if parts else 'run'


def run_test(args: dict, platform='cpu'):
    run_simulation(platform=platform, **args)
    return 0, 0.0

if __name__ == '__main__':
    # 1D
    minerals = ['calcite', 'dolomite']#, 'magnesite']
    nx = 200
    n_obl_mult = 1
    co2_injection = 0.1
    max_ts = 1.e-3
    flash='phreeqc' # 'phreeqc' # 'reaktoro'
    database = 'phreeqc' # 'phreeqc' # 'pitzer' # 'supcrtbl'
    # of = f'output_1D_{nx}_' + '_'.join(minerals) + f'_{n_obl_mult}_{co2_injection}_ts_{max_ts}_{flash}_{database}'

    # phreeqc
    run_simulation(domain='1D', nx=nx, perm_poro='power_8', n_obl_mult=n_obl_mult, minerals=minerals,
                co2_injection=co2_injection, max_ts=max_ts, output=False, flash=flash, database=database)

    # reaktoro
    # minerals = ['calcite'] # , 'dolomite', 'magnesite']
    # flash='reaktoro'
    # database='phreeqc'
    # n_obl_mult = 1
    # run_simulation(domain='2D', nx=10, perm_poro='power_8', n_obl_mult=n_obl_mult, minerals=minerals,
    #             co2_injection=1.0, max_ts=max_ts, flash=flash, output=False, database=database)

    # 2D
    # run_simulation(domain='2D', nx=10, perm_poro='power_8', max_ts=1.5e-3)
    n_obl_mult = 9
    inj_rate = 1e-3
    nx = 50
    minerals = ['calcite']#, 'dolomite', 'magnesite']
    max_ts = 6.e-5 * 1e-4 / inj_rate
    max_ts = 1.e-6
    co2_injection = 0.1
    # run_simulation(domain='2D', nx=nx, output=True, max_ts=max_ts,
    #                 n_obl_mult=n_obl_mult,
    #                 interpolator='multilinear',
    #                 output_folder=f'output_2D_{nx}_' + '_'.join(minerals) + f'_{n_obl_mult}_{co2_injection}_ts_{max_ts}',
    #                 #mesh_filename='input/wedge.msh',
    #                 poro_filename='input/spherical_50_5.txt', #'input/wedge_0.009.txt',#'old_calculations/calcite_2D_50_100/spherical_50_5_1/porosity_8.txt',
    #                 minerals=minerals,
    #                 h2o_injection=1.1,
    #                 co2_injection=co2_injection,
    #                 #inj_rate=inj_rate,
    #                 perm_poro='power_8',
    #                 platform='cpu',
    #                 ni_dt_increase_cutoff=4,
    #                 ni_dt_decrease_cutoff=6,
    #                 n_good_ts=15,
    #                 report_timesteps=6 * [5e-6])

    # 3D
    # run_simulation(domain='3D', max_ts=2.e-3, output=False,
    #                mesh_filename='input/core_13k.msh', poro_filename='input/core_13k_0.02.txt')
    # run_simulation(domain='3D', max_ts=1.e-3, output=True,
    #                mesh_filename='input/core_60k.msh', poro_filename='input/core_60k_0.01.txt')
    # run_simulation(domain='3D', max_ts=8.e-4, output=True, perm_poro='power_8',
    #                n_obl_mult=3, platform='cpu', minerals=['calcite'],
    #                mesh_filename='input/core_195k.msh', poro_filename='input/core_195k.txt')


# paths = ['./100x100/data_ts3.vts',
    #          './100x100/data_ts14.vts']
    # write_2d_output_for_paper(paths=paths)
    # paths = ['output_200/log.txt', 'output_2000_50000/log.txt',
    #          'output_200_1000_5/log.txt', 'output_500/log.txt',
    #          'output_200_1000_no_reaction/log.txt']
    # labels = [r'$n_x=200,\, \Delta t_{max}=10^{-3}$ day, $n_{obl}=5001$',
    #           r'$n_x=200,\, \Delta t_{max}=1\cdot 10^{-4}$ day, $n_{obl}=50001$',
    #           r'$n_x=200,\, \Delta t_{max}=5\cdot 10^{-3}$ day, $n_{obl}=1001$',
    #           r'$n_x=500,\, \Delta t_{max}=5\cdot 10^{-4}$ day \, $n_{obl}=5001$',
    #           r'$n_x=200,\, \Delta t_{max}=8\cdot 10^{-6}$ day, $n_{obl}=5001$, no reaction']
    # linestyle = ['-', '-.', ':', '-', '--']
    # colors=['b', 'b', 'b', 'r', 'b']
    # nx = [200, 200, 200, 500, 200]
    # plot_max_cfl(paths=paths, labels=labels, nx=nx, linestyle=linestyle, colors=colors)
