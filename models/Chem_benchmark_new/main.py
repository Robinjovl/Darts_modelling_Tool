from model import Model
from darts.engines import value_vector, redirect_darts_output

GRAV = '_grav'
grid_1D = 0
for res in [1]:
    redirect_darts_output('run' + str(res) + '.log')
    n = Model(grid_1D=grid_1D, res=res, custom_physics=0)
    n.init()
    n.set_output()
    n.params.max_ts = 1e-0

    n.run(50)
    # n.save_restart_data()
    n.print_timers()
    n.print_stat()

    # do not plot in pipelines. Do it only when debug it locally 
    if grid_1D:
       n.print_and_plot_1D()
    else:
       n.print_and_plot_2D()
