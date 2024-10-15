
from examples.case_1 import input_data_case_1
from examples.case_1_burden import input_data_case_1_burden_O1, input_data_case_1_burden_O2
from examples.case_1_burden import input_data_case_1_burden_U1, input_data_case_1_burden_U2
from examples.case_1_burden import input_data_case_1_burden_O1_U1, input_data_case_1_burden_O2_U2
from examples.case_2 import input_data_case_2
from examples.case_3 import input_data_case_3
from examples.case_4 import input_data_case_4
from examples.case_4 import input_data_case_4_no_conduction
from examples.case_4 import input_data_case_4_small_capacity
from examples.case_5 import input_data_case_5
from examples.whitby import input_data_case_whitby

def set_input_data(case: str):
    if case == 'case_1':
        input_data = input_data_case_1()
    elif case == 'case_1_burden_O1':
        input_data = input_data_case_1_burden_O1()
    elif case == 'case_1_burden_O2':
        input_data = input_data_case_1_burden_O2()
    elif case == 'case_1_burden_U1':
        input_data = input_data_case_1_burden_U1()
    elif case == 'case_1_burden_U2':
        input_data = input_data_case_1_burden_U2()
    elif case == 'case_1_burden_O1_U1':
        input_data = input_data_case_1_burden_O1_U1()
    elif case == 'case_1_burden_O2_U2':
        input_data = input_data_case_1_burden_O2_U2()
    elif case == 'case_2':
        input_data = input_data_case_2()
    elif case == 'case_3':
        input_data = input_data_case_3()
    elif case == 'case_4':
        input_data = input_data_case_4()
    elif case == 'case_4_no_conduction':
        input_data = input_data_case_4_no_conduction()
    elif case == 'case_4_small_capacity':
        input_data = input_data_case_4_small_capacity()
    elif case == 'case_5':
        input_data = input_data_case_5()
    elif case == 'whitby':
        input_data = input_data_case_whitby()
    else:
        assert False, f'Wrong case {case}'
    return input_data