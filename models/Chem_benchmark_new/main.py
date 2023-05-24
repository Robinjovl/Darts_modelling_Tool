from model import Model
from darts.engines import value_vector, redirect_darts_output

GRAV = '_grav'
grid_1D = 1
for res in [1]:
    redirect_darts_output('run' + str(res) + '.log')
    n = Model(grid_1D=grid_1D, res=res)
    n.init()
    n.params.max_ts = 1e-0

    n.run_python(50)
    n.save_restart_data()
    n.print_timers()
    n.print_stat()

    if grid_1D:
        n.print_and_plot_1D()
    else:
        n.print_and_plot_2D()
