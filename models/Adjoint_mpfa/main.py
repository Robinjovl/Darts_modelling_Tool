from adjoint_definition import prepare_synthetic_observation_data, read_observation_data, process_adjoint

# this adjoint gradient test is based on 2ph_comp model
if __name__ == '__main__':
    try:
        # if compiled with OpenMP, set to run with 1 thread, as MPFA tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    prepare_synthetic_observation_data()
    read_observation_data()
    failed = process_adjoint()
    print('----------------The status is: %s' % failed)