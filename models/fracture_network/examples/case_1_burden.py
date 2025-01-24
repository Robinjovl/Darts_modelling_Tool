import os
from .case_1 import input_data_case_1

def input_data_case_1_burden_O1_U1():
    idata = input_data_case_1()
    idata.geom['case_name'] = 'case_1_burden_O1_U1'

    # overburden layers (with fractures)
    idata.geom['overburden_thickness'] = idata.geom['height_res'] * 2
    idata.geom['overburden_layers'] = 1
    idata.geom['underburden_thickness'] = idata.geom['height_res'] * 2
    idata.geom['underburden_layers'] = 1

    return idata

def input_data_case_1_burden_O2_U2():
    idata = input_data_case_1_burden_O1_U1()
    idata.geom['case_name'] = 'case_1_burden_O2_U2'

    # second overburden layers (without fractures)
    idata.geom['overburden_2_thickness'] = idata.geom['height_res'] * 5
    idata.geom['overburden_2_layers'] = 1
    idata.geom['underburden_2_thickness'] = idata.geom['height_res'] * 5
    idata.geom['underburden_2_layers'] = 1

    return idata

def input_data_case_1_burden_O1():
    idata = input_data_case_1()
    idata.geom['case_name'] = 'case_1_burden_O1'

    idata.geom['overburden_thickness'] = idata.geom['height_res'] * 2
    idata.geom['overburden_layers'] = 1

    return idata

def input_data_case_1_burden_O2():
    idata = input_data_case_1_burden_O1()
    idata.geom['case_name'] = 'case_1_burden_O2'

    idata.geom['overburden_thickness'] = idata.geom['height_res'] * 2
    idata.geom['overburden_layers'] = 1

    # second overburden layers (without fractures)
    idata.geom['overburden_2_thickness'] = idata.geom['height_res'] * 5
    idata.geom['overburden_2_layers'] = 1

    return idata

def input_data_case_1_burden_U1():
    idata = input_data_case_1()
    idata.geom['case_name'] = 'case_1_burden_U1'

    idata.geom['underburden_thickness'] = idata.geom['height_res'] * 2
    idata.geom['underburden_layers'] = 1

    return idata

def input_data_case_1_burden_U2():
    idata = input_data_case_1_burden_U1()
    idata.geom['case_name'] = 'case_1_burden_U2'

    idata.geom['underburden_thickness'] = idata.geom['height_res'] * 2
    idata.geom['underburden_layers'] = 1

    # second overburden layers (without fractures)
    idata.geom['underburden_2_thickness'] = idata.geom['height_res'] * 5
    idata.geom['underburden_2_layers'] = 1

    return idata