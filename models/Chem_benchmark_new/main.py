from model import Model
from darts.engines import value_vector, redirect_darts_output

# Guard the entry point. The model enables parallel operator evaluation in init(),
# which creates a multiprocessing pool. Under the 'spawn' start method (Windows,
# macOS) every worker re-imports this module, so the simulation must run only when
# the file is executed as a script — otherwise each worker would recursively spawn
# more workers. See docs/for_developers/parallel_operators.md.
if __name__ == '__main__':
    GRAV = '_grav'
    grid_1D = 0
    for res in [1]:
        redirect_darts_output('run' + str(res) + '.log')
        n = Model(grid_1D=grid_1D, res=res, custom_physics=0)
        n.init()
        n.set_output()
        n.ts_control.dt_max = 1e-0

        n.run(50)
        # n.save_restart_data()
        n.print_timers()
        n.print_stat()

        # do not plot in pipelines. Do it only when debug it locally
        if grid_1D:
            n.print_and_plot_1D()
        else:
            n.print_and_plot_2D()
