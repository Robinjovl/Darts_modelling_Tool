from examples.base import InputDataConfig, build_input_data


def input_data_case_1():
    config = InputDataConfig(
        permeability=1000.0,
        rsv_top=1800.0,
        rsv_bottom=1900.0,
        rsv_xy=1500.0,
        doublet_shift=100.0,
        cell_shift=500.0,
        matrix_tags=(1, 2, 3),
        pressure_reference_depth=0.0,
    )
    idata = build_input_data(config)
    return idata
