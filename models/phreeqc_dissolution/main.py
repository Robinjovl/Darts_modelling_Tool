import os
os.environ["OMP_NUM_THREADS"] = "4"
import shutil
from model import Model
from darts.engines import redirect_darts_output
import numpy as np
from visualization import plot_profiles, plot_new_profiles, animate_1d

def run_simulation(domain: str, max_ts: float, nx: int = 100, mesh_filename: str = None, poro_filename: str = None, 
                   output: bool = False, interpolator: str = 'multilinear', minerals: list = ['calcite'], 
                   kinetic_mechanisms: list = ['acidic', 'neutral', 'carbonate'], output_folder: str = None,
                   n_obl_mult: int = 1, co2_injection: float = 0.1, platform: str = 'cpu'):
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
              co2_injection=co2_injection)

    # Initialize model
    m.init(itor_type=interpolator, platform=platform)
    m.physics.engine.n_solid = len(minerals)
    m.set_output(output_folder=output_folder, sol_filename=f'nx{nx}.h5')

    # Initialization check
    if platform == 'cpu':
        op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
        poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]
        volume = np.array(m.reservoir.mesh.volume, copy=False)
        total_pv = np.sum(volume[:m.n_res_blocks] * poro) * 1e6
        print(f'Total pore volume: {total_pv} cm3')
        print(f'Injection rate: {m.inj_cells.size} cells * {m.inj_rate / total_pv * 1e+6} PV/day')

    # visualization
    def plot(m, ith_step=None):
        if output:
            if domain == '1D': return plot_profiles(m, output_folder=output_folder)
            else: m.output.output_to_vtk(ith_step=ith_step)

    m.data_ts.dt_max = max_ts
    ith_step = 0
    if domain == '1D':
        fig_paths = []
        fig_paths.append(plot(m))
        m.run(days=0.002, restart_dt=max_ts)
        fig_paths.append(plot(m))
        m.run(days=0.018, restart_dt=max_ts)
        fig_paths.append(plot(m))
        m.run(days=0.02, restart_dt=max_ts)
        fig_paths.append(plot(m))
        m.data_ts.dt_max *= 3
        m.data_ts.first_ts = m.data_ts.dt_max
        m.run(days=0.1, restart_dt=max_ts)
        m.data_ts.dt_max *= 4
        m.run(days=0.86)
        fig_paths.append(plot(m))
        m.data_ts.dt_max *= 5
        m.data_ts.first_ts = m.data_ts.dt_max

        for i in range(7):
            dt = 2.0
            m.run(days=dt)
            if i < 1:
                m.data_ts.dt_max *= 1.5
                m.data_ts.first_ts = m.data_ts.dt_max
            fig_paths.append(plot(m))

        # if output:
        #     animate_1d(output_folder=output_folder, fig_paths=fig_paths)
    elif domain == '2D':
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001, restart_dt=max_ts)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.data_ts.dt_max *= 40
        m.data_ts.first_ts = m.data_ts.dt_max
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1

        for i in range(15):
            dt = 0.4
            m.run(days=dt)
            if i < 1:
                m.data_ts.dt_max *= 1.5
                m.data_ts.first_ts = m.data_ts.dt_max
            plot(m=m, ith_step=ith_step)
            ith_step += 1
    elif domain == '3D':
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001, restart_dt=max_ts)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.data_ts.dt_max *= 20
        m.data_ts.first_ts = m.data_ts.dt_max
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1
        m.run(days=0.001)
        plot(m=m, ith_step=ith_step)
        ith_step += 1

        for i in range(15):
            dt = 0.4
            m.run(days=dt)
            if i < 1:
                m.data_ts.dt_max *= 1.5
                m.data_ts.first_ts = m.data_ts.dt_max
            plot(m=m, ith_step=ith_step)
            ith_step += 1

    # Print some statistics
    print('\nNegative composition occurrence:', m.physics.reservoir_operators[0].counter, '\n')

    m.print_timers()
    m.print_stat()

    # copy files to save configuration
    # shutil.copy('main.py', os.path.join(output_folder, 'main.py'))
    # shutil.copy('model.py', os.path.join(output_folder, 'model.py'))

if __name__ == '__main__':
    # 1D
    run_simulation(domain='1D', nx=200, max_ts=1.e-3)
    # n_obl_mult = 1
    # run_simulation(domain='2D', nx=200, output=True, max_ts=1.e-3,
    #                 n_obl_mult=n_obl_mult,
    #                 interpolator='multilinear',
    #                 minerals=['calcite'],# 'dolomite'],#, 'magnesite'])  # 4.e-5)
    #                 kinetic_mechanisms=['acidic', 'neutral', 'carbonate'],
    #                 co2_injection=0.1,
    #                 platform='cpu')

    # 2D
    # run_simulation(domain='2D', nx=10, max_ts=2.e-3)
    # run_simulation(domain='2D', nx=100, max_ts=8.e-4, output=True, poro_filename='input/spherical_100_8.txt')

    # 3D
    # run_simulation(domain='3D', max_ts=2.e-3, output=False,
    #                mesh_filename='input/core_13k.msh', poro_filename='input/core_13k_0.02.txt')
    # run_simulation(domain='3D', max_ts=1.e-3, output=True,
    #                mesh_filename='input/core_60k.msh', poro_filename='input/core_60k_0.01.txt')
    # run_simulation(domain='3D', max_ts=8.e-4, output=True,
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
