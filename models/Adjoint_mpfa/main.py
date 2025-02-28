from adjoint_definition import prepare_synthetic_observation_data, read_observation_data, process_adjoint
from darts.models.cicd_model import get_platform, is_iter_solvers, set_one_thread

# this adjoint gradient test is based on 2ph_comp model
if __name__ == '__main__':
    set_one_thread()
    platform = get_platform()

    prepare_synthetic_observation_data()
    read_observation_data()
    failed = process_adjoint()
    print('----------------The status is: %s' % failed)
    exit(failed)