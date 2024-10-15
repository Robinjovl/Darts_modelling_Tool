import os
from .case_1 import input_data_case_1

def input_data_case_1_burden_O1_U1():
    input_data = input_data_case_1()
    input_data['case_name'] = 'case_1_burden_O1_U1'

    # overburden layers (with fractures)
    input_data['overburden_thickness'] = input_data['height_res'] * 2
    input_data['overburden_layers'] = 1
    input_data['underburden_thickness'] = input_data['height_res'] * 2
    input_data['underburden_layers'] = 1

    return input_data
def input_data_case_1_burden_O2_U2():
    input_data = input_data_case_1_burden_O1_U1()
    input_data['case_name'] = 'case_1_burden_O2_U2'

    # second overburden layers (without fractures)
    input_data['overburden_2_thickness'] = input_data['height_res'] * 5
    input_data['overburden_2_layers'] = 1
    input_data['underburden_2_thickness'] = input_data['height_res'] * 5
    input_data['underburden_2_layers'] = 1

    return input_data
def input_data_case_1_burden_O1():
    input_data = input_data_case_1()
    input_data['case_name'] = 'case_1_burden_O1'

    input_data['overburden_thickness'] = input_data['height_res'] * 2
    input_data['overburden_layers'] = 1

    return input_data
def input_data_case_1_burden_O2():
    input_data = input_data_case_1_burden_O1()
    input_data['case_name'] = 'case_1_burden_O2'

    input_data['overburden_thickness'] = input_data['height_res'] * 2
    input_data['overburden_layers'] = 1

    # second overburden layers (without fractures)
    input_data['overburden_2_thickness'] = input_data['height_res'] * 5
    input_data['overburden_2_layers'] = 1

    return input_data

def input_data_case_1_burden_U1():
    input_data = input_data_case_1()
    input_data['case_name'] = 'case_1_burden_U1'

    input_data['underburden_thickness'] = input_data['height_res'] * 2
    input_data['underburden_layers'] = 1

    return input_data
def input_data_case_1_burden_U2():
    input_data = input_data_case_1_burden_U1()
    input_data['case_name'] = 'case_1_burden_U2'

    input_data['underburden_thickness'] = input_data['height_res'] * 2
    input_data['underburden_layers'] = 1

    # second overburden layers (without fractures)
    input_data['underburden_2_thickness'] = input_data['height_res'] * 5
    input_data['underburden_2_layers'] = 1

    return input_data