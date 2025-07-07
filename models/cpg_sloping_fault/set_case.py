from darts.input.input_data import InputData

from case_base import input_data_base
from case_generate_5x3x4 import input_data_case_5x3x4
from case_generate_51x51x1 import input_data_case_51x51x1
from case_generate_100x100x100 import input_data_case_100x100x100
from case_40x40x10 import input_data_case_40x40x10

def set_input_data(idata: InputData, case: str):
    if 'generate_5x3x4' in case:
        input_data_case_5x3x4(idata, case)
    elif 'generate_51x51x1' in case:
        input_data_case_51x51x1(idata, case)
    elif 'generate_100x100x100' in case:
        input_data_case_100x100x100(idata, case)
    elif '40x40x10' in case:
        input_data_case_40x40x10(idata, case)
    else:
        input_data_base(idata, case)

