from darts.models.darts_model import DartsModel

from model import Model
from darts.engines import *
import numpy as np
import meshio
import os
from math import fabs

try:
    # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
    from darts.engines import set_num_threads
    set_num_threads(1)
except:
    pass

def run_python(m, days=0, restart_dt=0, log_3d_body_path=0, init_step = False):
    if days:
        runtime = days
    else:
        runtime = m.runtime

    mult_dt = m.params.mult_ts
    max_dt = m.params.max_ts
    m.e = m.physics.engine

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

        converged = run_timestep_python(m, dt, t)

        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                   % (ts, t, dt, m.e.n_newton_last_dt, m.e.n_linear_last_dt))
            if not init_step:
                m.reservoir.write_to_vtk(m.output_directory, m.ith_step + 1, m.physics.engine, dt)
                m.ith_step += 1
                if m.ith_step > 1000:
                    os._exit()

            if m.physics.engine.n_newton_last_dt < 4:
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
            if dt / mult_dt > 1.e-8 / 86400:
                dt /= mult_dt

            if dt < 1.e-2 / 86400.0 and m.physics.engine.momentum_inertia == 0.0 and m.enable_dynamic_mode: # less than smth -> go to fully dynamic (implicit) stepping
                m.physics.engine.momentum_inertia = 2406.0
                dt = 5.e-4 / 86400 # 500 microseconds
                max_dt = 5.e-4 / 86400 # 500 microseconds
                m.physics.engine.active_linear_solver_id = 1
                print("Fully dynamic mode enabled!!!")

            print("Cut timestep to %.5e" % dt)

        max_slip_area = np.max(np.array(m.slip_area))
        if m.ith_step + 1 > m.max_newt_it_dynamic_mode and m.slip_area[-1] < 0.005 * max_slip_area and m.enable_dynamic_mode and \
                m.ith_step_ready_for_reinjection == 0:
            m.ith_step_ready_for_reinjection = m.ith_step

        # if m.ith_step + 1 > 500 and m.slip_area[-1] < 0.005 * max_slip_area and m.enable_dynamic_mode and \
        #         m.ith_step - m.ith_step_ready_for_reinjection > 500:
        #     m.physics.engine.momentum_inertia = 0.0
        #     dt = 0.001
        #     m.params.max_ts = max_dt = 0.005
        #     m.enable_dynamic_mode = False
        #     m.reservoir.wells[0].control = m.physics.new_rate_prod(0.0)
        #     #X = np.array(m.physics.engine.X, copy = False)
        #     m.reservoir.wells[1].control = m.physics.new_bhp_inj(m.reservoir.p_init[m.id_inj])
            # m.physics.engine.active_linear_solver_id = 0
        #     print("Fully dynamic mode disabled!!!")


    # update current engine time
    m.e.t = runtime

    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (m.e.stat.n_timesteps_total, m.e.stat.n_timesteps_wasted,
                                                        m.e.stat.n_newton_total, m.e.stat.n_newton_wasted,
                                                        m.e.stat.n_linear_total, m.e.stat.n_linear_wasted))
def run_timestep_python(m, dt, t):
    self = m
    max_newt = self.params.max_i_newton
    self.e.n_linear_last_dt = 0
    well_tolerance_coefficient = 1e2
    self.timer.node['simulation'].start()
    res_history = []

    for i in range(max_newt + 1):
        self.e.assemble_linear_system(dt)
        res = self.e.calc_newton_dev()
        self.e.dev_p = res[0]
        self.e.dev_u = res[1]
        if len(res) > 2 and res[2] == res[2]:       self.e.dev_g = res[2]
        else:                                       self.e.dev_g = 0.0
        res_history.append((res[0], res[1], res[2]))

        self.e.newton_residual_last_dt = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + self.e.dev_g ** 2)
        #self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'rp = ' + str(self.e.dev_p) + '\t' + 'ru = ' + str(self.e.dev_u) + '\t' + \
                    'rg = ' + str(self.e.dev_g) + '\t' + 'rwell = ' + str(self.e.well_residual_last_dt))
        self.e.n_newton_last_dt = i
        #  check tolerance if it converges
        if ((self.e.dev_p < self.params.tolerance_newton and
             self.e.dev_u < self.params.tolerance_newton and
             self.e.dev_g < self.params.tolerance_newton and
             self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
              or self.e.n_newton_last_dt == self.params.max_i_newton):
            if (i > 0):  # min_i_newton
                if i < max_newt:
                    converged = 1
                else:
                    converged = 0
                break
        if self.e.dev_g > m.cut_off_gap_residual:
            converged = 0
            print('Restart newton iterations due to exceed of contact residual cut-off exceeded!!!')
            break

        r_code = self.e.solve_linear_equation()
        self.timer.node["newton update"].start()
        self.e.apply_newton_update(dt)
        self.timer.node["newton update"].stop()
        if i < max_newt:
            converged = 1

    if not hasattr(m, 'slip_area'):
        m.slip_area = [0.0]

    areas = calc_slip_area(m)
    cur_area = sum(areas)
    print('slip area = ' + str(cur_area))
    # print(areas)
    if m.enable_dynamic_mode:
        if cur_area - m.slip_area[-1] > 4.2 * m.min_area:
            converged *= 0
        else:
            m.slip_area.append(cur_area)
            converged *= 1

    converged = self.e.post_newtonloop(dt, t, converged)

    self.timer.node['simulation'].stop()
    return converged
def calc_slip_area(m):
    areas = []
    dz = np.max(m.reservoir.unstr_discr.mesh_data.points[:,2]) - np.min(m.reservoir.unstr_discr.mesh_data.points[:,2])
    for contact in m.physics.engine.contacts:
        cell_ids = np.array(contact.cell_ids, copy=True)
        for i in range(cell_ids.size):
            if contact.states[i] == contact_state.SLIP:
                areas.append(m.reservoir.unstr_discr.faces[cell_ids[i]][4].area / dz)
    return areas
def get_output_folder(config={'mode': 'quasi_static', 'depletion': {'mode': 'uniform'}, 'friction_law': 'static'}):
    return 'sol_' + config['mode'] + '_' + config['depletion']['mode'] + '_' + config['friction_law']
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

    m.params.first_ts = 1.0
    run_python(m, 1.0, init_step=True)
    m.reinit(zero_conduction=True)
    m.physics.engine.dt1 = 0.0
    m.physics.engine.find_equilibrium = False

    if m.depletion_mode == 'uniform':
        # no fluid flow, no mechanics -> flow coupling, keeping pressure -> mechanics influencing
        m.reservoir.apply_geomechanics_mode(physics=m.physics, mode=2)
    else:
        # eliminate mechanics -> flow coupling, keeping flow -> mechanics
        m.reservoir.apply_geomechanics_mode(physics=m.physics, mode=0)

    ## timestepping
    m.physics.engine.t = 0.0
    time = 0
    for ith_step, dt in enumerate(t):
        time += dt
        m.params.max_ts = dt
        m.params.mult_ts = 10.0
        run_python(m, dt)
        ith_step += 1

    m.print_timers()
    m.print_stat()

    labels = ['DARTS: ' + config['friction_law']]
    plot_analytics = config['friction_law'] if plot_analytics else None
    animate = True if len(t) > 1 else False
    plot_profiles(data_folder=m.output_directory, labels=labels, analytics=plot_analytics, animate=animate)

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
def plot_profiles(data_folder: str, labels: list, analytics=None, animate: bool=False):
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
            # substitute with your own path to FFMPEG installation
            rcParams['animation.ffmpeg_path'] = r'c:\work\packages\ffmpeg-6.0\bin\ffmpeg.exe'

            times, files = read_pvd(os.path.join(data_folder, 'solution_fault.pvd'))
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
                    time_text.set_text('time = ' + str(days) + ' day ' + str(minutes) + ' min ' + str(msec) + ' msec')

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

            anim = FuncAnimation(fig, animate, interval=2000, frames=np.arange(max_nt))
            writervideo = animation.FFMpegWriter(fps=2)
            video_filename = os.path.join(data_folder, 'fault_video.mp4')
            anim.save(video_filename, writer=writervideo)
        except:
            print('Cannot do the animation! Skipped. Check ffmeg is installed:', rcParams['animation.ffmpeg_path'])
            animate = False
    if not animate:
        pic_filename = os.path.join(data_folder, 'fault_plot.png')
        fig.savefig(pic_filename)
    plt.close(fig)
    # plt.show()

if __name__ == '__main__':
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
