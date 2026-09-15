from darts.models.darts_model import DartsModel
from darts.nonlinear_solvers.mechanics import configure_time_integration
from darts.engines import time_integration

from model import Model
from darts.engines import *
import glob
import sys
import numpy as np
import meshio
import os
from math import fabs

try:
    # one thread by default (the regression references were generated single-threaded; the multithreaded
    # engine_pm_cpu assembly is bit-identical to it since the race fixes, so config['n_threads'] may raise it)
    from darts.engines import set_num_threads
    set_num_threads(1)
except:
    pass

def back_to_quasi_static(m, reason):
    """End the co-seismic stage: switch the inertia term off, restore the selected time-integration scheme,
    re-inject the quasi-static linear solver and restart the timestep from `dt_after_arrest` (days). Used both
    on a natural rupture arrest and when the dynamic stage can no longer make progress at the timestep floor.
    Returns the new timestep."""
    engine = m.physics.engine
    engine.momentum_inertia = 0.0
    engine.dt1 = 0.0
    if getattr(m, '_be_fallback', False):
        engine.time_integration = m._saved_scheme
        m._be_fallback = False
    m.n_arrests = getattr(m, 'n_arrests', 0) + 1
    m.n_dynamic_steps = 0
    m.solver_phase = 'static'
    if m.linear_solver.open_source_solvers_available():
        qs_spec = getattr(m, 'stage_solver_specs', {}).get('quasi_static')
        if qs_spec is not None:
            m.linear_solver.update_solver(spec=qs_spec)
        else:
            m.linear_solver.update_solver(tolerance=1.e-10, max_iterations=500)
    else:
        engine.active_linear_solver_id = 0
    dt = getattr(m, 'dt_after_arrest', 1.e-3)
    print("Fully dynamic mode disabled (%s): back to quasi-static with dt = %.3e days" % (reason, dt))
    return dt


def run_python(m, days=0, restart_dt=0, log_3d_body_path=0, init_step = False):
    if days:
        runtime = days
    else:
        runtime = m.ts_control.runtime

    mult_dt = m.ts_control.dt_mult
    max_dt = m.ts_control.dt_max
    m.e = m.physics.engine
    # dynamic stage: below this dt a failing Newmark-family step is redone with backward Euler (see below)
    dt_dyn_fallback = getattr(m, 'dt_dyn_fallback', 0.2 * 5.e-4 / 86400)

    # get current engine time
    t = m.e.t

    # same logic as in engine.run
    if init_step:
        dt = days
    else:
        dt = min(max_dt, days)

    # evaluate end time
    runtime += t
    ts = 0
    m.ith_step_ready_for_reinjection = 0

    while t < runtime:
        if init_step:   new_time = t
        else:           new_time = t + dt

        if not init_step:
            m.timer.node["update"].start()
            # store boundaries taken at previous time step
            m.reservoir.update(dt=dt, time=new_time)
            if m.depletion_mode == 'uniform':
                m.update_pressure(dt=dt, time=new_time)
            m.reservoir.update_trans(dt, m.physics.engine.X)
            m.timer.node["update"].stop()

        converged = m.nonlinear_solver.run_timestep(dt, t)

        if converged:
            if getattr(m, '_be_fallback', False):
                m.physics.engine.time_integration = m._saved_scheme
                m._be_fallback = False
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                   % (ts, t, dt, m.nonlinear_solver.status.n_newton, m.nonlinear_solver.status.n_linear))
            if not init_step:
                dynamic = m.physics.engine.momentum_inertia > 0.0
                if dynamic:
                    m.n_dynamic_steps = getattr(m, 'n_dynamic_steps', 0) + 1
                vtk_every = getattr(m, 'vtk_every_dynamic', 1) if dynamic else 1
                if vtk_every > 0 and (m.ith_step + 1) % vtk_every == 0:
                    m.reservoir.write_to_vtk(m.output_directory, m.ith_step + 1, m.physics.engine, dt)
                m.ith_step += 1
                step_callback = getattr(m, 'step_callback', None)
                if step_callback is not None:
                    step_callback(m, t, dt, dynamic)
                if m.ith_step > getattr(m, 'max_steps', 1000):
                    exit(0)
                if dynamic and m.n_dynamic_steps >= getattr(m, 'max_dynamic_steps', np.inf):
                    print('Reached max_dynamic_steps = %d, stopping the run' % m.n_dynamic_steps)
                    m.stop_requested = True
                    break

            if m.nonlinear_solver.status.n_newton < 4:
                dt *= 1.5
            if dt > max_dt:
               dt = max_dt

            if t + dt > runtime:
               dt = runtime - t
        elif not m.enable_dynamic_mode:
            print("No converged solution found!")
            exit(-1)
        else:
            new_time -= dt
            engine = m.physics.engine
            dynamic = engine.momentum_inertia > 0.0
            if dynamic and not getattr(m, '_be_fallback', False) and dt < dt_dyn_fallback and \
                    engine.time_integration != time_integration.BACKWARD_EULER:
                # A velocity-state scheme (Newmark / generalized-alpha / Bathe) cannot be relaxed by cutting dt:
                # the free-flight motion u^n + dt v^n is imposed at any dt, so a stick/slip Newton limit cycle of the
                # contact return mapping at the rupture front persists (and below ~1e-6 s the 1/(beta dt^2)
                # amplification of displacement round-off floors the momentum residual). Backward Euler freezes the
                # matrix for small dt and converges, so the failing step is redone with backward Euler (its
                # velocity/acceleration state is carried on), after which the selected scheme is restored.
                m._be_fallback = True
                m._saved_scheme = engine.time_integration
                engine.time_integration = time_integration.BACKWARD_EULER
                m.n_be_fallbacks = getattr(m, 'n_be_fallbacks', 0) + 1
                print("Dynamic step failed with the %s scheme at dt = %.3e s: retrying with backward Euler"
                      % (str(m._saved_scheme).split('.')[-1], dt * 86400.0))
            elif dt / mult_dt > 1.e-8 / 86400:
                dt /= mult_dt
            elif dynamic and m.enable_dynamic_mode:
                # The timestep floor is reached and the dynamic step still fails, so the loop would retry the
                # same dt forever. Below dt ~ 1e-8 s the inertia term rho V / dt^2 amplifies the round-off of
                # the displacements above the momentum tolerance, so no smaller timestep can converge -- this
                # is the stick/slip chatter of the contact return mapping at the rupture front, and with
                # backward Euler there is no further fallback scheme. Treat it as the end of the co-seismic
                # stage: switch the inertia off and continue quasi-statically, which removes the 1/dt^2
                # amplification. Counted as `n_dynamic_stalls` (a natural arrest leaves it at zero).
                m.n_dynamic_stalls = getattr(m, 'n_dynamic_stalls', 0) + 1
                dt = back_to_quasi_static(
                    m, 'dynamic stage stalled at the timestep floor dt = %.3e s, slip area %.4f'
                       % (dt * 86400.0, m.slip_area[-1] if len(getattr(m, 'slip_area', [])) else 0.0))
                max_dt = m.ts_control.dt_max
                if t + dt > runtime:
                    dt = runtime - t

            if dt < 1.e-2 / 86400.0 and m.physics.engine.momentum_inertia == 0.0 and m.enable_dynamic_mode: # less than smth -> go to fully dynamic (implicit) stepping
                m.physics.engine.momentum_inertia = 2406.0
                # time integration of the inertia term (velocity/acceleration state starts from rest)
                ti = dict(m.time_integration)
                configure_time_integration(m.physics.engine, ti.pop('scheme', 'backward_euler'), **ti)
                print('Time integration: ' + str(m.time_integration))
                dt = 5.e-4 / 86400 # 500 microseconds
                max_dt = 5.e-4 / 86400 # 500 microseconds
                m.solver_phase = 'dynamic'
                if m.linear_solver.open_source_solvers_available():
                    # Dynamic (inertial) stage: switch to the stage solver (rebuilt
                    # and re-injected against the existing Jacobian -- a CPU or GPU
                    # registry spec, e.g. cuDSS for the quasi-static stage and
                    # GMRES+FS-CPR here; by default the quasi-static solver rebuilt
                    # at tolerance 1e-12, which also refreshes the FS-CPR
                    # displacement-block AMG for the inertial Jacobian). With
                    # config['linear_solver']['dynamic'] = 'inplace' the live stack is
                    # only tightened -- the open-source equivalent of the proprietary
                    # ls_params[1] switch (cpu_gmres_ilu0, tol 1e-12, 500 iters).
                    # update_solver() never touches the Jacobian.
                    dyn_spec = getattr(m, 'stage_solver_specs', {}).get('dynamic')
                    if dyn_spec is not None:
                        m.linear_solver.update_solver(spec=dyn_spec)
                    else:
                        m.linear_solver.update_solver(tolerance=1.e-12, max_iterations=500)
                else:
                    # Proprietary build: legacy engine-side solver bank switch.
                    m.physics.engine.active_linear_solver_id = 1
                print("Fully dynamic mode enabled!!!")

            print("Cut timestep to %.5e" % dt)

        max_slip_area = np.max(np.array(m.slip_area))
        if m.ith_step + 1 > m.max_newt_it_dynamic_mode and m.slip_area[-1] < 0.005 * max_slip_area and m.enable_dynamic_mode and \
                m.ith_step_ready_for_reinjection == 0:
            m.ith_step_ready_for_reinjection = m.ith_step

        # Rupture arrest -> back to the quasi-static stage (thesis Sec. 6.3): once the dynamic stage has run
        # for at least `min_dynamic_steps` steps and the slipping area has dropped below `arrest_area_fraction`
        # of its peak, the inertia term is switched off, the quasi-static solver is re-injected and the timestep
        # restarts from `dt_after_arrest` (days) with the schedule's dt_max; production continues. A later
        # loss of convergence re-enters the dynamic stage through the criterion above.
        if converged and m.physics.engine.momentum_inertia > 0.0 and m.enable_dynamic_mode and \
                getattr(m, 'n_dynamic_steps', 0) >= getattr(m, 'min_dynamic_steps', m.max_newt_it_dynamic_mode) and \
                m.slip_area[-1] < getattr(m, 'arrest_area_fraction', 0.005) * max_slip_area:
            dt = back_to_quasi_static(m, 'rupture arrested, slip area %.4f of peak %.4f'
                                      % (m.slip_area[-1], max_slip_area))
            max_dt = m.ts_control.dt_max
            if t + dt > runtime:
                dt = runtime - t


    # update current engine time
    m.e.t = runtime

    stats = m.nonlinear_solver.stats
    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (stats.n_timesteps_total, stats.n_timesteps_wasted,
                                                        stats.n_newton_total, stats.n_newton_wasted,
                                                        stats.n_linear_total, stats.n_linear_wasted))
def get_output_folder(config={'mode': 'quasi_static', 'depletion': {'mode': 'uniform'}, 'friction_law': 'static'}):
    name = 'sol_' + config['mode'] + '_' + config['depletion']['mode'] + '_' + config['friction_law']
    ti = config.get('time_integration')
    if ti and ti.get('scheme', 'backward_euler') != 'backward_euler':
        name += '_' + '_'.join(str(v) for v in ti.values())
    return name
def run_and_plot(config: dict, plot_analytics: bool=False, compare_with_ref=False):
    t = config['timesteps']

    ## model setup
    m = Model(config=config)
    m.init()
    m.output_directory = get_output_folder(config)
    redirect_darts_output(os.path.join(m.output_directory, 'log.txt'))
    m.timer.node["update"] = timer_node()
    m.ith_step = 0  # Store initial conditions as ../solution0.vtk

    # calculate fault cell size, for controlling timesteps during dynamic rupture propagation by limiting slip area increase
    m.min_area = 1.e10
    for contact in m.physics.engine.contacts:
        cell_ids = np.array(contact.cell_ids, copy=True)
        for i in range(cell_ids.size):
            m.min_area = min(m.min_area, m.reservoir.unstr_discr.faces[cell_ids[i]][4].area)
    m.min_area /= np.max(m.reservoir.unstr_discr.mesh_data.points[:,2]) - np.min(m.reservoir.unstr_discr.mesh_data.points[:,2])
    print('Min area = ' + str(m.min_area))

    # control maximum contact residual
    if m.enable_dynamic_mode:
        m.cut_off_gap_residual = 0.01# if self.e.momentum_inertia else 0.01
    else:
        m.cut_off_gap_residual = 100.0

    # optional run controls: VTK cadence in the dynamic stage (3D + fault files every n-th step; 1 = every step),
    # the step cap (the loop exits at max_steps), the arrest criterion (min_dynamic_steps, arrest_area_fraction,
    # dt_after_arrest) and OpenMP threads for the assembly / CPU solvers
    m.vtk_every_dynamic = config.get('vtk_every_dynamic', 1)
    m.max_steps = config.get('max_steps', 1000)
    for key in ('min_dynamic_steps', 'arrest_area_fraction', 'dt_after_arrest', 'max_dynamic_steps'):
        if key in config:
            setattr(m, key, config[key])
    if 'n_threads' in config:
        from darts.engines import set_num_threads
        set_num_threads(int(config['n_threads']))

    ## initialization
    # find equilibrium
    m.reservoir.set_equilibrium()
    m.physics.engine.find_equilibrium = True
    m.physics.engine.print_linear_system = False
    m.physics.engine.scale_rows = True
    m.physics.engine.scale_dimless = False
    # scaled unknowns
    # m.physics.engine.x_dim = 1.e-6
    # m.physics.engine.p_dim = 1.0
    # m.physics.engine.t_dim = 1.0
    # m.physics.engine.m_dim = 1.0

    m.ts_control.dt_first = 1.0
    run_python(m, 1.0, init_step=True)
    m.reinit(zero_conduction=True)
    m.physics.engine.dt1 = 0.0
    m.physics.engine.find_equilibrium = False

    if m.depletion_mode == 'uniform':
        # no fluid flow
        # no mechanics -> flow coupling, keeping pressure -> mechanics influencing
        m.reservoir.apply_geomechanics_mode(physics=m.physics, mode=2)
    else:
        # flud flow persists,
        # no mechanics -> flow coupling, keeping flow -> mechanics
        m.reservoir.apply_geomechanics_mode(physics=m.physics, mode=0)

    ## timestepping
    m.physics.engine.t = 0.0
    time = 0
    for ith_step, dt in enumerate(t):
        time += dt
        m.ts_control.dt_max = dt
        m.ts_control.dt_mult = 10.0
        run_python(m, dt)
        ith_step += 1
        if getattr(m, 'stop_requested', False):
            break

    m.print_timers()
    m.print_stat()

    labels = ['DARTS: ' + config['friction_law']]
    plot_analytics = config['friction_law'] if plot_analytics else None
    animate = True if len(t) > 1 else False
    plot_profiles(data_folder=m.output_directory, labels=labels, analytics=plot_analytics, animate=animate,
                  fps=config.get('animation_fps', 2), frame_stride=config.get('animation_stride', 1))

    ret_flag = 0
    if compare_with_ref:
        ret_flag = compare_solution_with_ref(m)
    return ret_flag


def compare_solution_with_ref(m : DartsModel, verbose = True):
    ith_step = 1  # compare only the last timestep result
    ith_step = str(ith_step)

    vtk_fname = 'solution_fault' + ith_step + '.vtu'  # a filename to read and compare (fault data)
    vtk_ref_fname = os.path.join(os.path.join('ref', m.output_directory), vtk_fname)
    vtk_cur_fname = os.path.join(m.output_directory, vtk_fname)

    props=['f_local', 'g_local', 'mu', 'p']  # property list (need for printing purposes)

    ref = read_vtk(vtk_ref_fname, props)  # the reference solution
    cur = read_vtk(vtk_cur_fname, props)  # the current solution
    names = ['centers', 'cell_data', 'points', 'point_data']  # object names to be compared

    rel_diff_tolerance = 1e-6
    abs_diff_tolerance = 1e-8
    eps_div = 1e-15  # to avoid division by zero
    ret_flag = 0
    for n, r, c in zip(names, ref, cur):
        if type(r) == dict: # cell_data is a dict, so check each item there
            if len(r) == 0:  # point_data is empty, skip it
                continue
            ns, rs, cs = r.keys(), r.values(), c.values()  # dict to list
        else:
            ns, rs, cs =  [n], [r], [c]  # create a list just to have a loop below for both cases
        for ni, ri, ci in zip(ns, rs, cs):
            r1 = np.array(ri)
            c1 = np.array(ci)
            diff = np.fabs(r1 - c1) / (np.fabs(r1) + eps_div) # relative difference
            diff_max = diff.max()
            if np.isclose(r1, c1, rtol=rel_diff_tolerance, atol=abs_diff_tolerance).all():
                if verbose:
                    print('Comparing', ni, 'diff', diff_max)
            else:
                ret_flag = 1
                print('There is a rel.difference', diff_max, 'for', ni)
    print('compare:', 'OK' if ret_flag == 0 else 'FAILED')
    return ret_flag

def run_test(args: dict, platform='cpu'):
    return run_and_plot(config=args, compare_with_ref=True), 0.0

def read_pvd(filename):
    from xml.dom.minidom import parse
    document = parse(filename)
    elems = document.getElementsByTagName('DataSet')
    timesteps = []
    files = []
    for step in elems:
        timesteps.append(float(step.getAttribute('timestep')))
        files.append(step.getAttribute('file'))
    return timesteps, files
def read_vtk(filename, props):
    import meshio

    mesh = meshio.read(filename=filename)

    # cell data
    centers = np.empty([0, 3])
    cell_data = {}
    for geom_name, geom in mesh.cells_dict.items():
        centers = np.append(centers, np.average(mesh.points[geom], axis=1), axis=0)
        for prop in props:
            if prop in mesh.cell_data_dict:
                if prop not in cell_data: cell_data[prop] = []
                cell_data[prop].append(mesh.cell_data_dict[prop][geom_name])

    # point data
    points = mesh.points
    point_data = {}
    for prop_name, prop in mesh.point_data.items():
        if prop_name in props:
            point_data[prop_name] = prop

    return centers, cell_data, points, point_data
def find_ffmpeg():
    """Path of an ffmpeg executable: FFMPEG_PATH, PATH, or the known Windows / conda locations (None if absent)."""
    import shutil
    candidates = [os.environ.get('FFMPEG_PATH'), shutil.which('ffmpeg'),
                  r'C:\software\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe', r'c:\work\packages\ffmpeg-6.0\bin\ffmpeg.exe']
    candidates += sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(sys.executable)), '..', '*', 'bin', 'ffmpeg')))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def plot_profiles(data_folder: str, labels: list, analytics=None, animate: bool=False, fps: int=2, frame_stride: int=1):
    """Fault profiles (slip, Coulomb / shear / normal stress, friction, pressure) from the fault VTK output;
    animate=True writes fault_video.mp4 over every frame_stride-th snapshot of solution_fault.pvd at the given fps
    (a GIF through Pillow when no ffmpeg is found), else fault_plot.png of solution_fault1.vtu."""
    from matplotlib import pyplot as plt
    ls = 13
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=ls)

    b1 = 2250 - 150
    b2 = 2250 + 150
    a1 = 2250 - 75
    a2 = 2250 + 75

    marker = ['', '', '', '', '', '', '', '', '', '', '', '', '']
    colors = ['b', 'r', 'g']
    linestyles = ['-', '-', '-']
    lw = 1
    msec0 = 0

    if animate:
        datafile = [os.path.join(data_folder, 'solution_fault0.vtu')]
    else:
        datafile = [os.path.join(data_folder, 'solution_fault1.vtu')]

    n_plots = 6
    fig, stress = plt.subplots(nrows=1, ncols=n_plots, sharey=True, figsize=(18, 8))
    for k, filename in enumerate(datafile):
        c, fault_data, __, __ = read_vtk(filename=filename, props=['f_local', 'g_local', 'mu', 'p'])
        # times, files = readPVD(dirs[k] + '/solution_fault.pvd')
        # days = int(times[file_id])
        # hours = int(24 * times[file_id]) - 24 * days
        # minutes = int(24 * 60 * times[file_id]) - 60 * (hours + 24 * days)
        # msec = int(86400 * 1000 * times[file_id]) - 86400 * 1000 * days - 60000 * minutes - msec0
        # if k == id_start_count_time:
        #     msec0 = msec
        #     msec = 0

        # label = 'time = ' + str(days) + ' day ' + str(minutes) + ' min ' + str(msec) + ' msec'
        # label = 'time = ' + str(round(hours, 2)) + ' hrs + ' + str(msec) + ' msec'
        # if msec > 1000:
        #     new_hours = int(msec / 1000 / 3600)
        #     new_minutes = (msec / 1000 / 60) - 60 * new_hours
        #     label = str(new_hours) + 'h ' + postfixes[k]
        # else:
        #     label = str(msec) + ' msec' + postfixes[k]

        if len(labels) > 0:
            label = labels[k]

        ids = np.argsort(c[:,1])
        c[:, 1] = 2250 - c[:, 1]
        l_slip, = stress[0].plot(np.abs(fault_data['g_local'][0][ids,1]) * 1e+3, c[ids,1], linewidth=lw, color=colors[k], linestyle=linestyles[k], marker=marker[k], markersize=1, label=label)
        #stress[1].plot(fault_data['g_local'][0][:,0], c[ids,1], linewidth=1, color=colors[k], linestyle=linestyles[k], marker=marker[k], markersize=1, label=labels[k])
        l_shear, = stress[2].plot(fault_data['f_local'][0][ids,1] / 10, c[ids,1], linewidth=lw, color=colors[k], linestyle=linestyles[k], marker=marker[k], markersize=1)
        l_normal, = stress[3].plot(-fault_data['f_local'][0][ids,0] / 10, c[ids,1], linewidth=lw, color=colors[k], linestyle=linestyles[k], marker=marker[k], markersize=1)
        #stress[4].plot(fault_data['mu'][0][ids], c[ids,1], linewidth=1, color=colors[k], linestyle=linestyles[k], marker=marker[k], markersize=1, label=labels[k])
        #l_pres, = stress[4].plot(fault_data['p'][0][ids] / 10, c[ids,1], linewidth=1, color=colors[k], linestyle=linestyles[k], marker=marker[k], markersize=1, label=labels[k])
        l_fric, = stress[4].plot(fault_data['mu'][0][ids], c[ids, 1], linewidth=lw, color=colors[k],
                                 linestyle=linestyles[k], marker=marker[k], markersize=1)
        l_pres, = stress[5].plot(fault_data['p'][0][ids] / 10, c[ids,1], linewidth=lw, color=colors[k], linestyle=linestyles[k], marker=marker[k], markersize=1)

        #p_lims[0] = fault_data['p'][0].min() if fault_data['p'][0].min() < p_lims[0] else p_lims[0]
        #p_lims[1] = fault_data['p'][0].max() if fault_data['p'][0].max() > p_lims[1] else p_lims[1]

        # Coulomb stress
        coulomb_stress = np.sqrt(fault_data['f_local'][0][:,1] ** 2 + fault_data['f_local'][0][:,2] ** 2) - \
                         fault_data['mu'][0] * np.fabs(fault_data['f_local'][0][:,0])
        l_coul, = stress[1].plot(coulomb_stress[ids] / 10, c[ids, 1], linewidth=lw, color=colors[k], linestyle=linestyles[k],
                       marker=marker[k], markersize=1)

        # Uenishi & Rice nucleation length
        # if len(dc) > 0:
        #     G = 65000
        #     nu = 0.15
        #     ids_nuc = np.argwhere(np.abs(fault_data['g_local'][0][:,1]) > 1.e-5)
        #     Wmean = np.mean((mu_s[k] - mu_d[k]) * np.abs(fault_data['f_local'][0][ids_nuc,0]))
        #     Lur = 1.158 * G * dc[k] / (1 - nu) / Wmean
        #     print('Lnuc ' + label + ' ' + str(Lur))

        lines = [l_slip, l_coul, l_shear, l_normal, l_fric, l_pres]

    if analytics is not None:
        import pandas as pd

        static_names = {'y1': '9 & 10 lefty_25', 'coulomb': '9 & 10 leftSigma_C_post_25',
                        'y2': '9 & 10 righty_25', 'slip': '9 & 10 rightdelta_25'}
        slip_weakening_names = {'y1': '14 lefty', 'coulomb': '14 leftSigma_C_post',
                                'y2': '14 righty', 'slip': '14 rightdelta'}

        if analytics == 'static':
            names = static_names
        elif analytics == 'slip_weakening':
            names = slip_weakening_names

        df = pd.read_excel(os.path.join('data', 'Data_GGGG_NovikovEtAl2024_corrected.xlsx'))
        df['identifier'] = df.iloc[:, 0].astype(str) + df.iloc[:, 1].astype(str)

        analytical_data = {}
        for key, val in names.items():
            row = df[df['identifier'] == val].iloc[0, 3:-1].dropna()
            analytical_data[key] = pd.to_numeric(row, errors='coerce').to_numpy(dtype=float)

        k = 0
        l_an_slip, = stress[0].plot(analytical_data['slip'] * 1e+3, 2250 - analytical_data['y2'], linewidth=lw, color=colors[k],
                             linestyle='--', marker=marker[k], markersize=1, label='Analytics')
        l_an_coul, = stress[1].plot(analytical_data['coulomb'] / 1e+6, 2250 - analytical_data['y1'], linewidth=lw, color=colors[k],
                                 linestyle='--', marker=marker[k], markersize=1)

    stress[0].set_ylabel(r'depth, $y$, m', fontsize=20)
    # legend_title = 'time = ' + str(days) + 'd ' + str(hours) + 'hrs ' + str(minutes) + 'min + '
    updated_legend = False
    if updated_legend:
        # new legend
        # Getting the handles and labels from stress[0]
        handles, labels = stress[0].get_legend_handles_labels()

        # Inserting a title in between items
        title_index = len(labels)//3  # Adjust this index to place the title where you want
        labels.insert(title_index, legend_title)

        # Creating a dummy line with no markers or line
        dummy_line = Line2D([0], [0], marker='none', color='none', linestyle='none', linewidth=0)
        handles.insert(title_index, dummy_line)

        # Now you create a legend with the modified handles and labels
        stress[0].legend(handles, labels, loc='upper left', prop={'size': 14})
    else:
        # current legend
        legend = stress[0].legend(loc='upper left', prop={'size': ls})
        if len(labels) == 0:
            legend.set_title(legend_title, prop={'size': ls})

    x_labels = [r'slip, mm', r'Coulomb stress, MPa', r'shear stress, MPa', r'effective normal stress, MPa', r'friction coefficient', r'pressure, MPa']#r'friction coefficient',
    fill_polys = []
    for i in range(n_plots):
        stress[i].axhline(y=b1, linestyle='--', color='k')
        stress[i].axhline(y=b2, linestyle='--', color='k')
        stress[i].axhline(y=a1, linestyle='--', color='k')
        stress[i].axhline(y=a2, linestyle='--', color='k')

        # stress[i].grid(True, which='both')
        alpha = 0.3
        stress[i].set_xlabel(x_labels[i], fontsize=15)
        # stress[i].set_ylim(list(stress[i].get_ylim()[::-1]))
        fill1 = stress[i].fill_between(x=[stress[i].set_xlim()[0], stress[i].set_xlim()[1]], y1=a1, y2=a2, color='palegoldenrod',
                             interpolate=True, alpha=alpha)
        fill2 = stress[i].fill_between(x=[stress[i].set_xlim()[0], stress[i].set_xlim()[1]], y1=b1, y2=a1, color='olive',
                             interpolate=True, alpha=alpha)
        fill3 = stress[i].fill_between(x=[stress[i].set_xlim()[0], stress[i].set_xlim()[1]], y1=a2, y2=b2, color='olive',
                             interpolate=True, alpha=alpha)
        fill_polys.append((fill1, fill2, fill3))

    stress[0].invert_yaxis()
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)

    if animate:
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import FuncAnimation
            from matplotlib import rcParams
            # ffmpeg: FFMPEG_PATH env var, PATH, or the known locations (download link
            # https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z); a Pillow GIF otherwise
            ffmpeg = find_ffmpeg()
            if ffmpeg:
                rcParams['animation.ffmpeg_path'] = ffmpeg
            times, files = read_pvd(os.path.join(data_folder, 'solution_fault.pvd'))
            times, files = times[::max(1, int(frame_stride))], files[::max(1, int(frame_stride))]
            max_nt = len(files)
            time_text = stress[0].text(0.07, 0.2, 'time = ' + str(24 * 60 * times[0]) + ' minutes', fontsize=12, rotation='horizontal', transform=fig.transFigure)

            def animate(i):
                nt = 50
                each_ith = 1 # int(max_nt / nt)
                if i % each_ith == 0:
                    c, fault_data, __, __ = read_vtk(filename=os.path.join(data_folder, files[i]),
                                                     props=['f_local', 'g_local', 'mu', 'p'])
                    ids = np.argsort(c[:, 1])
                    c[:, 1] = 2250 - c[:, 1]
                    # slip
                    lines[0].set_data(fault_data['g_local'][0][ids, 1] * 1e+3, c[ids, 1])
                    xmin = 1e+3 * np.min(fault_data['g_local'][0][ids, 1])
                    xmax = 1e+3 * np.max(fault_data['g_local'][0][ids, 1])
                    stress[0].set_xlim(xmin, xmax)
                    # Coulomb stress
                    coulomb_stress = np.sqrt(fault_data['f_local'][0][:, 1] ** 2 + fault_data['f_local'][0][:, 2] ** 2) - \
                                     fault_data['mu'][0] * np.fabs(fault_data['f_local'][0][:, 0])
                    lines[1].set_data(coulomb_stress[ids] / 10, c[ids, 1])
                    xmin = np.min(coulomb_stress[ids] / 10)
                    xmax = np.max(coulomb_stress[ids] / 10)
                    stress[1].set_xlim(-20, 0)  # (xmin, xmax)
                    # shear stress
                    lines[2].set_data(fault_data['f_local'][0][ids, 1] / 10, c[ids, 1])
                    xmin = np.min(fault_data['f_local'][0][ids, 1] / 10)
                    xmax = np.max(fault_data['f_local'][0][ids, 1] / 10)
                    stress[2].set_xlim(0, 25)  # (xmin, xmax)
                    # normal stress
                    lines[3].set_data(fault_data['f_local'][0][ids, 0] / 10, c[ids, 1])
                    xmin = np.min(fault_data['f_local'][0][ids, 0] / 10)
                    xmax = np.max(fault_data['f_local'][0][ids, 0] / 10)
                    stress[3].set_xlim(20, 45)  # (xmin, xmax)
                    # friction coefficient
                    lines[4].set_data(fault_data['mu'][0][ids], c[ids, 1])
                    xmin = np.min(fault_data['mu'][0][ids])
                    xmax = np.max(fault_data['mu'][0][ids])
                    stress[4].set_xlim(0.95 * xmin, 1.05 * xmax)
                    # pressure
                    lines[5].set_data(fault_data['p'][0][ids] / 10, c[ids, 1])
                    xmin = np.min(fault_data['p'][0][ids])
                    xmax = np.max(fault_data['p'][0][ids])
                    stress[5].set_xlim(0, 40)

                    days = int(times[i])
                    minutes = int(24 * 60 * times[i]) - 24 * 60 * days
                    msec = int(86400 * 1000 * times[i]) - 86400 * 1000 * days - 60000 * minutes
                    time_text.set_text('step=' + str(i) + ' time = ' + str(days) + ' day ' + str(minutes) + ' min ' + str(msec) + ' msec')

                    for i in range(n_plots):
                        depth_lims = stress[i].get_ylim()
                        stress[i].set_ylim([max(depth_lims), min(depth_lims)])

                        alpha = 0.3
                        stress[i].set_xlabel(x_labels[i], fontsize=15)
                        # stress[i].set_ylim(list(stress[i].get_ylim()[::-1]))

                        new_vertices = [[stress[i].set_xlim()[0], a1],
                                        [stress[i].set_xlim()[1], a1],
                                        [stress[i].set_xlim()[1], a2],
                                        [stress[i].set_xlim()[0], a2],
                                        [stress[i].set_xlim()[0], a1]]  # close the loop
                        fill_polys[i][0].set_paths([new_vertices])
                        new_vertices = [[stress[i].set_xlim()[0], b1],
                                        [stress[i].set_xlim()[1], b1],
                                        [stress[i].set_xlim()[1], a1],
                                        [stress[i].set_xlim()[0], a1],
                                        [stress[i].set_xlim()[0], b1]]  # close the loop
                        fill_polys[i][1].set_paths([new_vertices])
                        new_vertices = [[stress[i].set_xlim()[0], a2],
                                        [stress[i].set_xlim()[1], a2],
                                        [stress[i].set_xlim()[1], b2],
                                        [stress[i].set_xlim()[0], b2],
                                        [stress[i].set_xlim()[0], a2]]  # close the loop
                        fill_polys[i][2].set_paths([new_vertices])

                        # stress[i].fill_between(x=[stress[i].set_xlim()[0], stress[i].set_xlim()[1]], y1=a1, y2=a2,
                        #                        color='palegoldenrod',
                        #                        interpolate=True, alpha=alpha)
                        # stress[i].fill_between(x=[stress[i].set_xlim()[0], stress[i].set_xlim()[1]], y1=b1, y2=a1,
                        #                        color='olive',
                        #                        interpolate=True, alpha=alpha)
                        # stress[i].fill_between(x=[stress[i].set_xlim()[0], stress[i].set_xlim()[1]], y1=a2, y2=b2,
                        #                        color='olive',
                        #                        interpolate=True, alpha=alpha)

                return lines  # not really necessary, but optional for blit algorithm

            anim = FuncAnimation(fig, animate, interval=int(1000 / fps), frames=np.arange(max_nt))
            if ffmpeg:
                writervideo = animation.FFMpegWriter(fps=fps)
                video_filename = os.path.join(data_folder, 'fault_video.mp4')
            else:
                writervideo = animation.PillowWriter(fps=fps)
                video_filename = os.path.join(data_folder, 'fault_video.gif')
            anim.save(video_filename, writer=writervideo)
            print('animation written:', video_filename, '(%d frames)' % max_nt)
        except Exception as ex:  # noqa: BLE001
            print('Cannot do the animation! Skipped (%s). Check ffmpeg is installed: %s' % (ex, rcParams.get('animation.ffmpeg_path')))
            animate = False
    if not animate:
        pic_filename = os.path.join(data_folder, 'fault_plot.png')
        fig.savefig(pic_filename)
    plt.close(fig)
    # plt.show()


def coseismic_snapshots(data_folder: str, gap_days: float = 1.e-4):
    """First co-seismic stage of a run from its fault output: (t0, [(t_days, file), ...]) where t0 is the time of
    the last quasi-static snapshot before nucleation (the list starts with it) and the rest are the snapshots
    closer together than gap_days (a dynamic step is at most 5e-4 s, a quasi-static one days). (None, []) if the
    run has not reached the dynamic stage."""
    times, files = read_pvd(os.path.join(data_folder, 'solution_fault.pvd'))
    first = next((i + 1 for i in range(len(times) - 1) if 0 < times[i + 1] - times[i] < gap_days), None)
    if first is None:
        return None, []
    snaps = [(times[first - 1], os.path.join(data_folder, files[first - 1]))]
    for i in range(first, len(times)):
        if times[i] - times[i - 1] >= gap_days:
            break  # back to quasi-static stepping after the arrest
        snaps.append((times[i], os.path.join(data_folder, files[i])))
    return times[first - 1], snaps


def plot_profiles_compare(data_folders: list, labels: list, out_file: str, fps: int = 5, frame_stride: int = 1,
                          snapshot_times=()):
    """plot_profiles for several runs at once: the same six panels (slip, Coulomb / shear / effective normal
    stress, friction coefficient, pressure against depth) with its fixed stress limits and reservoir bands, one
    colour per run (b, r, g, m, c, k), animated through the co-seismic stage. The runs are aligned on the start of
    their dynamic stage: the frames follow the co-seismic snapshots of the run that lasts longest (every
    frame_stride-th), every other run shows its snapshot nearest in time since its own start, and a run that has
    ended keeps its last one. Writes out_file (a GIF through Pillow when no ffmpeg is found) and, for each time in
    snapshot_times (seconds since the start of the dynamic stage), <out_file without extension>_<ms>ms.png of the
    nearest frame. plot_profiles itself is left untouched: the regression test draws through it."""
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import rcParams
    runs = []
    for folder, label in zip(data_folders, labels, strict=True):
        t0, snaps = coseismic_snapshots(folder)
        if t0 is None:
            print('%s: still quasi-static, no co-seismic snapshot yet -- left out' % folder)
        else:
            runs.append((label, t0, snaps))
    if not runs:
        raise ValueError('none of the runs has reached the co-seismic stage')

    # the configuration of plot_profiles
    ls = 13
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=ls)
    b1, b2, a1, a2 = 2250 - 150, 2250 + 150, 2250 - 75, 2250 + 75
    colors = ['b', 'r', 'g', 'm', 'c', 'k']
    lw = 1
    x_labels = [r'slip, mm', r'Coulomb stress, MPa', r'shear stress, MPa', r'effective normal stress, MPa',
                r'friction coefficient', r'pressure, MPa']
    n_plots = 6
    day = 86400.0

    def profiles(filename):
        c, fault_data, __, __ = read_vtk(filename=filename, props=['f_local', 'g_local', 'mu', 'p'])
        ids = np.argsort(c[:, 1])
        f, g = fault_data['f_local'][0][ids], fault_data['g_local'][0][ids]
        mu, p = fault_data['mu'][0][ids], fault_data['p'][0][ids]
        coulomb = np.sqrt(f[:, 1] ** 2 + f[:, 2] ** 2) - mu * np.fabs(f[:, 0])
        return 2250 - c[ids, 1], [g[:, 1] * 1e+3, coulomb / 10, f[:, 1] / 10, f[:, 0] / 10, mu, p / 10]

    fig, stress = plt.subplots(nrows=1, ncols=n_plots, sharey=True, figsize=(18, 8))
    run_lines = [[stress[j].plot([], [], linewidth=lw, color=colors[k], linestyle='-',
                                 label=run[0] if j == 0 else None)[0] for j in range(n_plots)]
                 for k, run in enumerate(runs)]
    depth = np.concatenate([profiles(run[2][0][1])[0] for run in runs])
    margin = 0.05 * (depth.max() - depth.min())
    stress[0].set_ylim(depth.max() + margin, depth.min() - margin)
    stress[0].set_ylabel(r'depth, $y$, m', fontsize=20)
    stress[0].legend(loc='upper left', prop={'size': ls})
    fill_polys = []
    for i in range(n_plots):
        for y in (b1, b2, a1, a2):
            stress[i].axhline(y=y, linestyle='--', color='k')
        stress[i].set_xlabel(x_labels[i], fontsize=15)
        x0, x1 = stress[i].get_xlim()
        fill_polys.append(tuple(stress[i].fill_between(x=[x0, x1], y1=ya, y2=yb, color=col, interpolate=True, alpha=0.3)
                                for ya, yb, col in ((a1, a2, 'palegoldenrod'), (b1, a1, 'olive'), (a2, b2, 'olive'))))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    stress[0].set_zorder(1)  # the legend and the time label may reach over the next panel: keep them on top
    time_text = stress[0].text(0.07, 0.2, '', fontsize=12, rotation='horizontal', transform=fig.transFigure)

    longest = max(runs, key=lambda run: run[2][-1][0] - run[1])
    frame_t = [(t - longest[1]) * day for t, __ in longest[2]][::max(1, int(frame_stride))]

    def draw(i):
        ts = frame_t[i]
        slip, mu, ended = [], [], []
        for k, (label, t0, snaps) in enumerate(runs):
            __, f = min(snaps, key=lambda s: abs((s[0] - t0) * day - ts))
            if ts > (snaps[-1][0] - t0) * day + 1.e-9:
                ended.append(label)
            d, prof = profiles(f)
            for j in range(n_plots):
                run_lines[k][j].set_data(prof[j], d)
            slip.append(prof[0])
            mu.append(prof[4])
        slip, mu = np.concatenate(slip), np.concatenate(mu)
        limits = [(slip.min(), slip.max()), (-20, 0), (0, 25), (20, 45), (0.95 * mu.min(), 1.05 * mu.max()), (0, 40)]
        for j, (lo, hi) in enumerate(limits):
            if hi - lo < 1.e-9:
                lo, hi = lo - 1., hi + 1.
            stress[j].set_xlim(lo, hi)
            for poly, (ya, yb) in zip(fill_polys[j], ((a1, a2), (b1, a1), (a2, b2)), strict=True):
                poly.set_paths([[[lo, ya], [hi, ya], [hi, yb], [lo, yb], [lo, ya]]])
        time_text.set_text('step=%d time since nucleation = %.1f msec%s'
                           % (i, 1e3 * ts, ('\nended: ' + ', '.join(ended)) if ended else ''))
        return [ln for lines in run_lines for ln in lines]

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        rcParams['animation.ffmpeg_path'] = ffmpeg
    else:
        out_file = os.path.splitext(out_file)[0] + '.gif'
    anim = animation.FuncAnimation(fig, draw, interval=int(1000 / fps), frames=np.arange(len(frame_t)))
    anim.save(out_file, writer=animation.FFMpegWriter(fps=fps) if ffmpeg else animation.PillowWriter(fps=fps))
    print('animation written:', out_file, '(%d frames)' % len(frame_t))
    for ts in snapshot_times:
        i = int(np.argmin([abs(x - ts) for x in frame_t]))
        draw(i)
        png = '%s_%dms.png' % (os.path.splitext(out_file)[0], int(round(1e3 * frame_t[i])))
        fig.savefig(png)
        print('frame written:', png)
    plt.close(fig)


def run_tests():
    test_args_fault = []
    config = {'mode': 'quasi_static',
              'timesteps': [1.0],
              'depletion': {'mode': 'uniform', 'value': -250.0},
              'friction_law': 'static',
              'mesh_file': 'meshes/new_setup_coarse.geo',
              'cache_discretizer': False}
    # optional: Pardiso (pypardiso / Intel MKL) direct solve instead of the default GMRES + FS-CPR
    #config['use_pardiso'] = True
    config[0] = config['friction_law']  # to make work arg[0] in for_each_model
    test_args_fault += [config]
    config = {'mode': 'quasi_static',
              'timesteps': [1.0],
              'depletion': {'mode': 'uniform', 'value': -172.4},  # -172.685 is more precise, requires finer mesh
              'friction_law': 'slip_weakening',
              'mesh_file': 'meshes/new_setup_coarse.geo',
              'cache_discretizer': False}
    config[0] = config['friction_law']  # to make work arg[0] in for_each_model
    test_args_fault += [config]

    rcode = 0
    failed_tests = []
    for config_i in test_args_fault:
        rcode += run_test(args=config_i, platform='cpu')[0]
        if rcode != 0:
            failed_tests += [config_i]
    if rcode != 0:
        print('Failed tests configs: ', failed_tests)
        print('Tests failed:', len(failed_tests))
    else:
        print('All tests passed successfully!')

def run_all():
    cases = []

    config = {'mode': 'mixed',
              'timesteps': np.ones(25),
              'depletion': {'mode': 'uniform', 'value': -290.6 / 25},
              'friction_law': 'slip_weakening',
              'mesh_file': 'meshes/new_setup_coarse_longer.geo'}
    # commented because it is very long
    #cases += [config]

    config = {'mode': 'mixed',
              'timesteps': 5 * np.ones(4),
              'depletion': {'mode': 'well', 'value': -250.0},
              'friction_law': 'slip_weakening',
              'mesh_file': 'meshes/new_setup_coarse.geo'}
    #cases += [config]

    config = {'mode': 'quasi_static',
              'timesteps': [1.0],
              'depletion': {'mode': 'uniform', 'value': -250.0},
              'friction_law': 'static',
              'mesh_file': 'meshes/new_setup_coarse.geo'}
    cases += [config]

    config = {'mode': 'quasi_static',
              'timesteps': [1.0],
              'depletion': {'mode': 'uniform', 'value': -172.4}, # -172.685 is more precise, requires finer mesh
              'friction_law': 'slip_weakening',
              'mesh_file': 'meshes/new_setup_coarse.geo'}
    cases += [config]

    config = {'mode': 'quasi_static',
              'timesteps': 25 * [1.0],
              'depletion': {'mode': 'uniform', 'value': -290.6 / 25},
              'friction_law': 'rsf',
              'mesh_file': 'meshes/new_setup_rsf.geo'}
    # commented because it is very long
    # cases += [config]

    for case in cases:
        if case['mode'] == 'quasi_static' and (case['friction_law'] == 'static' or case['friction_law'] == 'slip_weakening'):
            plot_analytics = True
        else:
            plot_analytics = False
        run_and_plot(config=case, plot_analytics=plot_analytics)

    # labels = ['DARTS: slip_weakening']
    # output_directory = 'sol_mixed_uniform_slip_weakening'
    # plot_profiles(data_folder=output_directory, labels=labels, analytics=None, animate=True)

if __name__ == '__main__':
    run_tests()
    #run_all()
