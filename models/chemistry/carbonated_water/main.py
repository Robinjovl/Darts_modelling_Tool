import os, signal, sys
os.environ["OMP_NUM_THREADS"] = "16"
import shutil
from model import Model
from darts.engines import redirect_darts_output
import numpy as np
from visualization import plot_profiles, plot_new_profiles, animate_1d


def build_report_timesteps(segments):
    """
    Build a per-step dt array from (upper_cum, n_steps) segments.

    Each segment [lower, upper] is divided into n_steps equal substeps.
    lower is 0 for the first segment, and the previous upper thereafter.
    """
    dts = []
    lower = 0.0
    for upper, n in segments:
        dts.extend([(upper - lower) / n] * n)
        lower = upper
    return np.array(dts)


def run_simulation(domain: str, max_ts: float, nx: int = 100, mesh_filename: str = None, poro_filename: str = None,
                   output: bool = False, interpolator: str = 'multilinear', minerals: list = ['calcite'],
                   kinetic_mechanisms: list = ['acidic', 'neutral', 'carbonate'], output_folder: str = None,
                   n_obl_mult: int = 1, co2_injection: float = 0.1, h2o_injection: float = 1.1,
                   inj_rate: float = None, perm_poro: str = 'power_8', platform: str = 'cpu',
                   ni_dt_increase_cutoff: int = 5, ni_dt_decrease_cutoff: int = 8, n_good_ts: int = 10, report_timesteps = None,
                   flash: str = 'phreeqc', database: str = 'phreeqc',
                   parallel_evaluation: bool = False, n_workers: int = None):
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

    m.verbose = m.VERBOSE_TIMERS

    # Initialize model
    m.init(itor_type=interpolator, platform=platform, n_solid=len(minerals),
           parallel_evaluation=parallel_evaluation, n_workers=n_workers)

    # Persistent OBL-axis box (MR327-equivalent correct_obl_axes): every Newton
    # update is clamped into the parametrization region on CPU and GPU, which
    # prevents excursions that trigger PHREEQC dilution fallbacks, wasted
    # iterations and unbounded adaptive-cache growth.
    from darts.engines import value_vector
    m.physics.engine.correct_obl_axes(value_vector(list(map(float, m.axes_min))),
                                      value_vector(list(map(float, m.axes_max))))
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
        if getattr(m.physics, 'cache', False):
            try:
                m.physics.write_cache()
            except Exception as exc:
                print(f"OBL cache flush on SIGTERM failed: {exc}")
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    try:
        m.n_good_ts = n_good_ts
        m.ni_dt_increase_cutoff = ni_dt_increase_cutoff
        m.ni_dt_decrease_cutoff = ni_dt_decrease_cutoff

        # initialization without injection
        if minerals == ['calcite']:
            init_days = 0.1
            num_time_iterations = 7
        elif set(minerals) == set(['calcite', 'dolomite']):
            init_days = 20.0
            num_time_iterations = 7
        else:
            init_days = 365.0
            num_time_iterations = 7

        rate = m.inj_rate
        m.inj_rate = 0.0

        m.ts_control.dt_max = 5.0
        _n_good_ts_saved = m.n_good_ts
        m.n_good_ts = 10**18      # disable dt_max growth -> hard 5-day ceiling during init
        m.run(days=init_days)
        m.n_good_ts = _n_good_ts_saved

        # injection
        m.inj_rate = rate
        m.physics.engine.t = 0.0
        ith_step = 0
        m.ts_control.dt_max = max_ts
        if domain == '1D':
            m.ts_control.dt_first = 1.e-6
            m.ts_control.dt_mult = 1.5
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
            # m.ts_control.dt_max *= 3
            m.ts_control.first_ts = m.ts_control.dt_max
            m.run(days=0.1, restart_dt=max_ts)
            # m.ts_control.dt_max *= 4
            m.run(days=0.86)
            fig_paths.append(plot(m))
            # m.ts_control.dt_max *= 5
            m.ts_control.first_ts = m.ts_control.dt_max

            for i in range(num_time_iterations):
                dt = 2.0
                m.run(days=dt)
                if i < 1:
                    # m.ts_control.dt_max *= 1.5
                    m.ts_control.first_ts = m.ts_control.dt_max
                fig_paths.append(plot(m))

            # if output:
            #     animate_1d(output_folder=output_folder, fig_paths=fig_paths)
        elif domain in ('2D', '3D'):
            plot(m=m, ith_step=ith_step)
            ith_step += 1

            if report_timesteps is None:
                # (upper_cum, n_steps) — extra refinement applied only in [1e-2, 1e-1]
                segments = [
                    (0.001, 1),
                    (0.005, 2),
                    (0.010, 2),
                    (0.030, 2),
                    (0.050, 2),
                    (0.100, 2),
                    (0.300, 2),
                    (0.500, 2),
                ]
                base = build_report_timesteps(segments)
                if domain == '2D':
                    report_timesteps = base * 1e-4 / m.inj_rate
                    n_fine = 6
                    dt_max_bump = 1.0
                else:  # 3D
                    report_timesteps = base * 0.0016128 / m.inj_rate
                    n_fine = 6
                    dt_max_bump = 30.0
                default_run = True
            else:
                report_timesteps = np.asarray(report_timesteps)
                n_fine = len(report_timesteps)
                dt_max_bump = 1.0
                default_run = False

            m.ts_control.dt_first = m.prev_dt = min(1.e-6 * 1e-3 / m.inj_rate, m.ts_control.dt_max)
            m.ts_control.dt_mult = 1.5
            ts_after_bt = 0
            max_snapshots_after_bt = 5
            for i, rts in enumerate(report_timesteps):
                if i == n_fine and dt_max_bump != 1.0:
                    m.ts_control.dt_max *= dt_max_bump
                    m.ts_control.first_ts = m.ts_control.dt_max
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


def run_1d(output: bool=False, n_obl_mult: int=1, platform: str='cpu',
            output_folder: str=None, database: str='phreeqc', flash: str='phreeqc',
            minerals: list=['calcite'], nx: int=200, two_phase_flag: bool=False,
            inj_rate: float=None):
    # inj_rate default None -> Model.set_physics computes the current 1D rate
    # self.volume * 24 == 0.1 * 0.007 * 0.008415 * 24 ~= 1.4137e-4 m3/day
    # (see model.py:248). Pass a float to override.

    max_ts = 1.e-3

    if two_phase_flag:
        # increase CO2 content in injected water
        co2_injection = 1.1
    else:
        co2_injection = 0.1

    # use supcrtbl/reaktoro database/solver for 3-mineral carbonate system.
    if set(minerals) == {'calcite', 'dolomite', 'magnesite'}:
        database = 'supcrtbl'
        flash = 'reaktoro'
        n_obl_mult = 9

    if output_folder is None:
        output_folder = f'output_1D_{nx}_' + '_'.join(minerals) + f'_{n_obl_mult}_{co2_injection}_ts_{max_ts}_{flash}_{database}'

    run_simulation(domain='1D', nx=nx, perm_poro='power_8', n_obl_mult=n_obl_mult,
                minerals=minerals, co2_injection=co2_injection, max_ts=max_ts,
                output=output, output_folder=output_folder, flash=flash, database=database,
                inj_rate=inj_rate,
                parallel_evaluation=True, platform=platform, n_workers=32)

def run_2d(output: bool=False, n_obl_mult: int=6, platform: str='cpu',
            output_folder: str=None, database: str='phreeqc', flash: str='phreeqc',
            minerals: list=['calcite'], two_phase_flag: bool=False,
            inj_rate: float=None, nx: int=25, poro_filename: str='input/spherical_25_2.txt'):

    if inj_rate is None:
        inj_rate = 1e-3
    max_ts = 6.e-5 * 1e-4 / inj_rate
    max_ts = 1.e-6

    if two_phase_flag:
        # increase CO2 content in injected water
        co2_injection = 1.1
        n_obl_mult = 6
    else:
        co2_injection = 0.1

    if output_folder is None:
        output_folder = f'output_2D_{nx}_' + '_'.join(minerals) + f'_{inj_rate}_{n_obl_mult}_{co2_injection}_ts_{max_ts}'
    run_simulation(domain='2D', nx=nx, output=True, max_ts=max_ts,
                    n_obl_mult=n_obl_mult,
                    output_folder=output_folder,
                    poro_filename='input/spherical_25_2.txt', #'input/wedge_0.009.txt',#'old_calculations/calcite_2D_50_100/spherical_50_5_1/porosity_8.txt',
                    minerals=minerals,
                    h2o_injection=1.1,
                    co2_injection=co2_injection,
                    inj_rate=inj_rate,
                    perm_poro='power_8',
                    platform=platform,
                    flash=flash,
                    database=database,
                    ni_dt_increase_cutoff=4,
                    ni_dt_decrease_cutoff=9,
                    n_good_ts=10,
                    parallel_evaluation=True,
                    n_workers=32)

def run_3d(case: str='13k', output: bool=False, n_obl_mult: int=3, platform: str='cpu',
            output_folder: str=None, database: str='phreeqc', flash: str='phreeqc',
            minerals: list=['calcite'], two_phase_flag: bool=False, inj_rate: float=None):


    if two_phase_flag:
        # increase CO2 content in injected water
        co2_injection = 1.1
        n_obl_mult = 6
    else:
        co2_injection = 0.1

    # use supcrtbl/reaktoro database/solver for 3-mineral carbonate system.
    if set(minerals) == {'calcite', 'dolomite', 'magnesite'}:
        database = 'supcrtbl'
        flash = 'reaktoro'
        n_obl_mult = 9

    if case == '13k':
        max_ts = 2.e-3
        poro_filename = 'input/core_13k_0.02.txt'
        mesh_filename = 'input/core_13k.msh'
    elif case == '60k':
        max_ts = 1.e-3
        poro_filename = 'input/core_60k_0.01.txt'
        mesh_filename = 'input/core_60k.msh'
    elif case == '195k':
        max_ts = 8.e-4
        poro_filename = 'input/core_195k.txt'
        mesh_filename = 'input/core_195k.msh'
        pass
    elif case == '474k':
        max_ts = 5.e-4
        pass
    else:
        raise NotImplementedError(f"{case} is not supported")

    if output_folder is None:
        of = f'output_3D_{case}_' + f'{platform}_' + '_'.join(minerals) + f'_{n_obl_mult}_{co2_injection}_ts_{max_ts}_{flash}_{database}'

    run_simulation(domain='3D',
                    max_ts=max_ts,
                    output=output,
                    output_folder=of,
                    n_obl_mult=n_obl_mult,
                    platform=platform,
                    mesh_filename=mesh_filename,
                    poro_filename=poro_filename,
                    minerals=minerals,
                    h2o_injection=1.1,
                    co2_injection=co2_injection,
                    # inj_rate=inj_rate,
                    perm_poro='power_8',
                    flash=flash,
                    database=database,
                    ni_dt_increase_cutoff=4,
                    ni_dt_decrease_cutoff=9,
                    n_good_ts=10,
                    parallel_evaluation=True,
                    n_workers=32)

if __name__ == '__main__':
    # 1D
    run_1d(two_phase_flag=False)
    # 2D
    # run_2d(platform='cpu')
    # 3D
    # run_3d()
