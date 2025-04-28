from adjoint_definition import prepare_synthetic_observation_data, read_observation_data, process_adjoint

# this adjoint gradient test is based on 2ph_comp model
if __name__ == '__main__':
    prepare_synthetic_observation_data()
    read_observation_data()
    failed = process_adjoint()
    print('----------------The status is: %s' % failed)