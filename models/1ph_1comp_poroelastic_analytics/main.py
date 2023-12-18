from model import Model, load_performance_data, check_performance_data
from darts.engines import *
import numpy as np
import meshio
from math import fabs
import os

from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams["text.usetex"]=False
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
font = {'family' : 'normal',
        'size'   : 18}
plt.rc('legend',fontsize=12)

def run_python(m, days=0, restart_dt=0, log_3d_body_path=0, init_step = False):
    if days:
        runtime = days
    else:
        runtime = m.runtime

    mult_dt = m.params.mult_ts
    max_dt = m.params.max_ts
    m.e = m.engine

    # get current engine time
    t = m.e.t

    # same logic as in engine.run
    if fabs(t) < 1e-15:
        dt = m.params.first_ts
    elif restart_dt > 0:
        dt = restart_dt
    else:
        dt = m.params.max_ts

    # evaluate end time
    runtime += t
    ts = 0

    while t < runtime:
        if init_step:   new_time = t
        else:           new_time = t + dt

        if not init_step:
            m.timer.node["update"].start()
            # store boundaries taken at previous time step
            m.reservoir.update(dt=dt, time=new_time)
            # evaluate and assign transient boundaries or sources / sinks
            if m.case == 'mandel':
                m.reservoir.update_mandel_boundary(dt=dt, time=new_time, physics=m.physics)
            # update transient boundaries or sources / sinks
            m.reservoir.update_trans(dt, m.engine.X)
            m.timer.node["update"].stop()

        converged = run_timestep_python(m, dt, t)
        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                   % (ts, t, dt, m.e.n_newton_last_dt, m.e.n_linear_last_dt))

            dt *= 1.5
            if dt > max_dt:
               dt = max_dt

            if t + dt > runtime:
               dt = runtime - t
        else:
            new_time -= dt
            dt /= mult_dt
            print("Cut timestep to %.5e" % dt)
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
    for i in range(max_newt + 1):
        self.e.run_single_newton_iteration(dt)
        res = self.e.calc_newton_dev()#self.e.calc_newton_residual()
        self.e.dev_p = res[0]
        self.e.dev_u = res[1]
        if len(res) > 2 and res[2] == res[2]:       self.e.dev_g = res[2]
        else:                                       self.e.dev_g = 0.0

        self.e.newton_residual_last_dt = np.sqrt(self.e.dev_u ** 2 + self.e.dev_p ** 2 + self.e.dev_g ** 2)
        #self.e.newton_residual_last_dt = self.e.calc_newton_residual()
        self.e.well_residual_last_dt = self.e.calc_well_residual()
        print(str(i) + ': ' + 'rp = ' + str(self.e.dev_p) + '\t' + 'ru = ' + str(self.e.dev_u) + '\t' + \
                    'rg = ' + str(self.e.dev_g) + '\t' + 'rwell = ' + str(self.e.well_residual_last_dt) + '\t' + 'CFL = ' + str(self.e.CFL_max))

        self.e.n_newton_last_dt = i
        #  check tolerance if it converges
        if ((self.e.dev_p < self.params.tolerance_newton and self.e.dev_u < self.params.tolerance_newton and self.e.dev_g < self.params.tolerance_newton
           and self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
              or self.e.n_newton_last_dt == self.params.max_i_newton):
            if (i > 0):  # min_i_newton
                if i < max_newt:
                    converged = 1
                else:
                    converged = 0
                break

        r_code = self.e.solve_linear_equation()
        self.timer.node["newton update"].start()
        self.e.apply_newton_update(dt)
        self.timer.node["newton update"].stop()
        if i < max_newt:
            converged = 1

    # End of newton loop
    converged = self.e.post_newtonloop(dt, t, converged)
    self.timer.node['simulation'].stop()
    return converged
def test(case='mandel', discr_name='mech_discretizer', mesh='rect', overwrite='0'):
    '''
    :param case: mandel/terzaghi
    :param scheme: stabilized/non_stabilized
    :param mesh: cells shape hex/rect/wedge
    :param overwrite: write pkl file even if it exists
    :return: tuple (bool failed, float64 time)
    '''
    print('case:' + case, 'discr_name:' + discr_name, 'mesh: ' + mesh, 'overwrite: ' + overwrite, sep=', ')
    import platform

    nt = 20
    max_dt = 200
    t = np.logspace(-3, np.log10(max_dt), nt)
    # nt = 200
    # max_t = 200
    # t = max_t / nt * np.ones(nt)

    m = Model(case=case, discretizer=discr_name, mesh=mesh)
    m.init()
    redirect_darts_output('log.txt')
    # redirect_darts_output('log.txt')
    # output_directory = 'sol_{:s}'.format(m.physics_type)
    m.timer.node["update"] = timer_node()
    m.engine.find_equilibrium = False

    time = 0.0
    data = []

    # poromech tests run with direct linear solvers (superlu), but somehow there is a difference
    # while using old and new lib. To handle this, use '_iter' pkls for old lib
    pkl_suffix = ''
    if os.getenv('ODLS') != None and os.getenv('ODLS') == '0':
        pkl_suffix = '_iter'
    file_name = os.path.join('ref', 'perf_' + case + '_' + discr_name + '_' + mesh + '_' +
                             platform.system().lower()[:3] + pkl_suffix + '.pkl')
    failed = 0

    is_plk_exist = os.path.isfile(file_name)
    if is_plk_exist:
        ref_data = load_performance_data(file_name=file_name)

    for ith_step, dt in enumerate(t):
        time += dt
        m.params.first_ts = dt
        m.params.max_ts = dt
        run_python(m, dt)

        # write a vtk snapshot
        # m.reservoir.write_to_vtk(output_directory, ith_step + 1, m.physics)
        data.append(m.get_performance_data(is_last_ts=(ith_step == t.size - 1)))
        if is_plk_exist:
            # to compare with analytic need only solution vector
            if False:
                an_data_step = {'solution' : get_analytic_solution(m, discr_name, t=time)['solution']}
                sol_data_step = {'solution' : get_solution_slice(m, discr_name, mesh, data[ith_step])['solution']}
                for k in ['reservoir blocks', 'variables']:
                    an_data_step[k] = sol_data[k] = data[ith_step][k]
            else:
                sol_data_step = data[ith_step]
                ref_data_step = ref_data[ith_step]
            failed += check_performance_data(ref_data_step, sol_data_step, failed,
                                             png_suffix=case+'_'+discr_name+'_'+mesh+'_'+str(ith_step))
            assert not failed

    if not is_plk_exist or overwrite == '1':
        m.save_performance_data(data=data, file_name=file_name)
        return False, 0.0
    # m.print_timers()

    if is_plk_exist:
        return (failed > 0), data[-1]['simulation time']
    else:
        return False, -1.0
def run_and_plot(case='mandel', discretizer='mech_discretizer'):
    ## only with rectangular mesh
    nt = 60
    max_dt = 30  # sec
    t = np.logspace(-3, np.log10(max_dt), nt)
    # nt = 200
    # max_t = 200
    # t = max_t / nt * np.ones(nt)

    # GeosX
    # t = np.empty(shape=(0,), dtype=np.float64)
    # t = np.append(t, 60 * np.ones(int((600 - 0) / 60)) / 86400)
    # t = np.append(t, 600 * np.ones(int((3600-600) / 600)) / 86400)
    # t = np.append(t, 3600 * np.ones(int((86400-3600) / 3600)) / 86400)
    # t = np.append(t, 86400 * np.ones(int((17280000-86400) / 86400)) / 86400)
    # nt = t.size

    m = Model(case=case, discretizer=discretizer)
    m.init()
    redirect_darts_output('log.txt')
    m.output_directory = 'sol_' + case + '_' + discretizer + '_rect'
    m.timer.node["update"] = timer_node()
    # m.physics.engine.find_equilibrium = False

    # for rectangular grid
    if discretizer == 'pm_discretizer':
        nx = np.unique(np.array([m.reservoir.unstr_discr.mat_cell_info_dict[i].centroid[0] for i in range(m.reservoir.unstr_discr.mat_cells_tot)]).round(decimals=4)).size
        ny = int(m.reservoir.unstr_discr.mat_cells_tot / nx)
        x = np.array([m.reservoir.unstr_discr.mat_cell_info_dict[i * ny].centroid[0] for i in range(nx)])
        xc = np.array([m.reservoir.unstr_discr.mat_cell_info_dict[i * ny].centroid for i in range(nx)])
    elif discretizer == 'mech_discretizer':
        xc = np.array([np.array(c.values) for c in m.reservoir.discr_mesh.centroids[:m.reservoir.n_matrix]])
        nx = np.unique(np.round(xc[:,0], decimals=6)).size
        ny = int(m.reservoir.n_matrix / nx)
        x = xc[::ny, 0]
        xc = xc[::ny]
    pres = { 'name': 'p', 'darts': np.zeros((nt + 1, nx)), 'analytics': np.zeros((nt + 1, nx)), 'time': np.zeros(nt + 1), 'x': x }
    disp = { 'name': 'u', 'darts': np.zeros((nt + 1, nx)), 'analytics': np.zeros((nt + 1, nx)), 'time': np.zeros(nt + 1), 'x': x }
    if case == 'mandel':
        pres['analytics'][0] = m.reservoir.mandel_exact_pressure(t=0.0, xc=x)
        p,ux,uy = m.reservoir.mandel_exact_displacements(t=0.0, xc=xc)
        disp['analytics'][0] = ux
    elif case == 'terzaghi':
        pres['analytics'][0] = m.reservoir.terzaghi_exact_pressure(t=0.0, xc=x)
        disp['analytics'][0] = m.reservoir.terzaghi_exact_displacements(t=0.0, xc=x)
    elif case == 'terzaghi_two_layers':
        pres['analytics'][0] = m.reservoir.terzaghi_two_layers_exact_pressure(t=0, xc=x)
        disp['analytics'][0] = m.reservoir.terzaghi_two_layers_exact_displacement(t=0, xc=x)

    time = 0.0
    for ith_step, dt in enumerate(t):
        time += dt
        m.params.first_ts = dt
        m.params.max_ts = dt
        run_python(m, dt)

        # save pressure
        X = np.array(m.engine.X, copy=False)
        pres['darts'][ith_step + 1] = X[m.engine.P_VAR::m.engine.N_VARS][::ny] # for rectangular grid
        disp['darts'][ith_step + 1] = X[m.engine.U_VAR::m.engine.N_VARS][::ny] # for rectangular grid
        if case == 'mandel':
            pres['analytics'][ith_step + 1] = m.reservoir.mandel_exact_pressure(t=time, xc=x)
            p,ux,uy = m.reservoir.mandel_exact_displacements(t=time, xc=xc)
            disp['analytics'][ith_step + 1] = ux
        elif case == 'terzaghi':
            pres['analytics'][ith_step + 1] = m.reservoir.terzaghi_exact_pressure(t=time, xc=x)
            disp['analytics'][ith_step + 1] = m.reservoir.terzaghi_exact_displacements(t=time, xc=x)
        elif case == 'terzaghi_two_layers':
            pres['analytics'][ith_step + 1] = m.reservoir.terzaghi_two_layers_exact_pressure(t=time, xc=x)
            disp['analytics'][ith_step + 1] = m.reservoir.terzaghi_two_layers_exact_displacement(t=time, xc=x)

        pres['time'][ith_step + 1] = time
        disp['time'][ith_step + 1] = time
        # write a vtk snapshot
        if discretizer == 'mech_discretizer':
            m.reservoir.write_to_vtk_mech_discretizer(m.output_directory, ith_step + 1, m.engine)
        elif discretizer == 'pm_discretizer':
            m.reservoir.write_to_vtk_pm_discretizer(m.output_directory, ith_step + 1, m.engine)
    m.print_timers()

    if case != 'terzaghi_two_layers_no_analytics':
        save_data = True
        plot_comparison(m, pres, discretizer, case, save_data=save_data)
        plot_comparison(m, disp, discretizer, case, save_data=save_data)
def plot_comparison(m, data, discretizer, case, save_data=False):
    prefix = m.output_directory
    tD, dataD = m.reservoir.tD, m.reservoir.pD
    name = data['name']
    if name == 'u':
        dataD = 1.0

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(15, 6))

    # initial pressure increase
    ax[0].axhline(y=data['analytics'][1,0] / dataD, linestyle='--', color='k')

    darts_name = 'DARTS ' + discretizer
    an_linestyle = '-'
    darts_linestyle = '--'
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    # pressure against time
    ax[0].semilogx(data['time'][1:] / tD, data['analytics'][1:,0] / dataD, color='b', linewidth=1, linestyle=an_linestyle, label='Analytics')
    ax[0].semilogx(data['time'][1:] / tD, data['darts'][1:,0] / dataD, color='b', linewidth=1, linestyle=darts_linestyle, label=darts_name)

    # pressure over domain
    nt = data['time'].size
    id_snaps = [1, int(nt / 2) + 1, int(0.8 * nt)]
    for i in range(len(id_snaps)):
        t_id = id_snaps[i]
        if name == 'p':
            ax[1].plot(data['x'], np.fabs(data['analytics'][t_id]) / dataD, color=colors[i], linestyle=an_linestyle, label=r'Analytics: $t = $' + '{:.2e}'.format(data['time'][t_id] / tD) + r' $t_D$')
        elif name == 'u':
            ax[1].plot(data['x'], data['analytics'][t_id] / dataD, color=colors[i], linestyle=an_linestyle,
                       label=r'Analytics: $t = $' + '{:.2e}'.format(data['time'][t_id] / tD) + r' $t_D$')
        ax[1].plot(data['x'], data['darts'][t_id] / dataD, color=colors[i], linestyle=darts_linestyle, label=darts_name + r': $t = $' + '{:.2e}'.format(data['time'][t_id] / tD) + r' $t_D$')

    if case == 'mandel':
        y_label = r'$2p\;/\;F$'
    elif case == 'terzaghi':
        y_label = r'$p\;/\;F$'
    elif case == 'terzaghi_two_layers':
        y_label = r'$p$'
    if name == 'u':
        y_label = r'$u$'

    ax[0].set_xlabel(r'$t\;/\;t_D$', fontsize=20)
    ax[0].set_ylabel(y_label, fontsize=20)
    ax[0].grid(True)
    ax[0].legend(loc='lower left', prop={'size': 14})

    ax[1].set_ylabel(y_label, fontsize=20)
    ax[1].set_xlabel(r'$x$', fontsize=20)
    ax[1].grid(True)
    ax[1].legend(loc='lower left', prop={'size': 14 }, framealpha=0.5)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    if name == 'p':
        plt.savefig(prefix + '/' + 'pressure_' + discretizer + '_' + case + '.png')
    elif name == 'u':
        plt.savefig(prefix + '/' + 'displacement_' + discretizer + '_' + case + '.png')
    # plt.show()

    if save_data:
        filename = prefix + '/' + name + '_data.txt'
        A = np.zeros((data['time'].shape[0], data['x'].shape[0], 2))
        A[:, :, 0] = data['time'][:, np.newaxis]
        A[:, :, 1] = data['x'][np.newaxis, :]
        np.savetxt(filename, np.c_[A[:,:,0].flatten(), A[:,:,1].flatten(), data['analytics'].flatten()])
def run(case='mandel', discretizer='mech_discretizer', mesh='rect'):
    if case == 'bai':
        nt = 100
        max_dt = 0.01
        t = np.logspace(-7, np.log10(max_dt), nt)
    else:
    nt = 60
        max_dt = 30
    t = np.logspace(-3, np.log10(max_dt), nt)

    m = Model(case=case, discretizer=discretizer, mesh=mesh)
    m.init()

    redirect_darts_output('log.txt')
    m.output_directory = 'sol_' + case + '_' + discretizer + '_' + mesh
    ith_step = 0
    m.timer.node["update"] = timer_node()
    # set equilibrium (including boundary conditions)
    # m.reservoir.set_equilibrium()
    # m.physics.engine.find_equilibrium = True
    # m.params.first_ts = 1
    # run_python(m, 1.0)
    # m.reinit_reference(output_directory)
    # m.physics.engine.find_equilibrium = False

    # m.engine.print_linear_system = True
    if discretizer == 'mech_discretizer':
        m.reservoir.write_to_vtk_mech_discretizer(m.output_directory, 0, m.engine)
    elif discretizer == 'pm_discretizer':
        m.reservoir.write_to_vtk_pm_discretizer(m.output_directory, 0, m.engine)
    time = 0.0
    for ith_step, dt in enumerate(t):
        time += dt
        m.params.first_ts = dt
        m.params.max_ts = dt
        run_python(m, dt)
        if discretizer == 'mech_discretizer':
            m.reservoir.write_to_vtk_mech_discretizer(m.output_directory, ith_step + 1, m.engine)
        elif discretizer == 'pm_discretizer':
            m.reservoir.write_to_vtk_pm_discretizer(m.output_directory, ith_step + 1, m.engine)

    m.print_timers()

def run_test(args: list = []):
    if len(args) > 3:
        return test(case=args[0], discr_name=args[1], mesh=args[2], overwrite=args[3])
    else:
        print('Not enough arguments provided')
        return 1, 0.0

# Rectangular grid, comparison to analytics
# run_and_plot(case='terzaghi', discretizer='mech_discretizer')
# run_and_plot(case='terzaghi', discretizer='pm_discretizer')
# run_and_plot(case='mandel', discretizer='mech_discretizer')
# run_and_plot(case='mandel', discretizer='pm_discretizer')
run(case='bai', discretizer='mech_discretizer')

# Wedge (triangular) grid
# ret = run(case='terzaghi', discretizer='mech_discretizer', mesh='wedge')
# run(case='terzaghi', discretizer='pm_discretizer', mesh='wedge')
# run(case='mandel', discretizer='mech_discretizer', mesh='wedge')
# run(case='mandel', discretizer='pm_discretizer', mesh='wedge')

# Unstructured hexahedral grid
# run(case='terzaghi', discretizer='mech_discretizer', mesh='hex')
# run(case='terzaghi', discretizer='pm_discretizer', mesh='hex')
# run(case='mandel', discretizer='mech_discretizer', mesh='hex')
# run(case='mandel', discretizer='pm_discretizer', mesh='hex')

def get_x(m, discr_name):
    # for rectangular grid
    if discr_name == 'pm_discretizer':
        nx = np.unique(np.array([m.reservoir.unstr_discr.mat_cell_info_dict[i].centroid[0] for i in range(m.reservoir.unstr_discr.mat_cells_tot)]).round(decimals=4)).size
        ny = int(m.reservoir.unstr_discr.mat_cells_tot / nx)
        x = np.array([m.reservoir.unstr_discr.mat_cell_info_dict[i * ny].centroid[0] for i in range(nx)])
        xc = np.array([m.reservoir.unstr_discr.mat_cell_info_dict[i * ny].centroid for i in range(nx)])
    elif discr_name == 'mech_discretizer':
        xc = np.array([np.array(c.values) for c in m.reservoir.discr_mesh.centroids[:m.reservoir.n_matrix]])
        nx = np.unique(np.round(xc[:,0], decimals=6)).size
        ny = m.reservoir.n_matrix // nx
        x = xc[::ny, 0]
        xc = xc[::ny]
    return nx, ny, x, xc

def get_analytic_solution(m, discr_name, t):
    '''
    :param m: Model
    :param discr_name: 'pm_discretizer' (poroelasticity) or 'mech_discretizer' (thermoporoelasticity)
    :param t: time (double)
    :return:
    '''
    # get analytic solution
    ref_data = {}
    ref_data['reservoir blocks'] = m.reservoir.mesh.n_blocks
    ref_data['variables'] = ['ux', 'uy', 'uz']
    ref_data['variables'].insert(m.reservoir.p_var, 'p')

    nx, ny, x, xc = get_x(m, discr_name)

    uy = uz = -999  # undefined
    if case == 'mandel':
        pressure = m.reservoir.mandel_exact_pressure(t=t, xc=x)  # 1D
        p,ux,uy = m.reservoir.mandel_exact_displacements(t=t, xc=xc)  # 2D
    elif case == 'terzaghi':
        pressure = m.reservoir.terzaghi_exact_pressure(t=t, xc=x)
        ux = m.reservoir.terzaghi_exact_displacements(t=t, xc=x)
    elif case == 'terzaghi_two_layers':
        pressure = m.reservoir.terzaghi_two_layers_exact_pressure(t=t, xc=x)
        ux = m.reservoir.terzaghi_two_layers_exact_displacement(t=t, xc=x)

    nvars = 4 # m.physics.n_vars #TODO why m.physics.n_vars == 1 ?
    ref_data['solution'] = np.zeros(nvars * nx)
    ref_data['solution'][m.reservoir.p_var::nvars] = pressure
    ref_data['solution'][m.reservoir.u_var::nvars] = ux
    ref_data['solution'][m.reservoir.u_var+1::nvars] = uy
    ref_data['solution'][m.reservoir.u_var+2::nvars] = uz
    return ref_data

def get_solution_slice(m, discr_name, mesh, sol_data):
    nx, ny, x, xc = get_x(m, discr_name)
    nvars = m.engine.N_VARS
    sol_data_slice = sol_data.copy()  # copy keys of the dictionary
    sol_data_slice['solution'] = np.zeros(nx * nvars)
    for v in range(nvars):
        if mesh == 'rect':
            sol_data_slice['solution'][v::nvars] = sol_data['solution'][v::nvars][::ny]
        else:
            assert False  #TODO implement for unstructured grid

    return sol_data_slice

# for case in ['terzaghi', 'mandel']: #TODO, 'terzaghi_two_layers']:
#     for mesh in ['rect']:#, 'wedge', 'hex']
#         if case == 'terzaghi_two_layers' and mesh == 'hex':
#             continue
#     mech_res = test(case=case, discr_name='mech_discretizer', mesh=mesh)
#     pm_res   = test(case=case, discr_name='pm_discretizer',   mesh=mesh)

