from model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.models.cicd_model import compare_solution_with_reference, get_platform


def run(platform='cpu'):

    GRAV = '_grav'
    grid_1D = True
    for res in [1]:
        redirect_darts_output('run' + str(res) + '.log')
        m = Model(grid_1D=grid_1D, res=res, custom_physics=0)
        m.init(platform=platform)
        m.set_output()
        m.params.max_ts = 1e-0

        m.run(50)
        # m.save_restart_data()
        m.save_data_to_h5('solution')
        m.print_timers()
        m.print_stat()

        # do not plot in pipelines. Do it only when debug it locally
        if grid_1D:
           m.print_and_plot_1D()
        else:
           m.print_and_plot_2D()

    # for CI/CD
    failed, sim_time = compare_solution_with_reference(m=m)
    return failed

if __name__ == '__main__':
    exit(run(platform=get_platform()))
