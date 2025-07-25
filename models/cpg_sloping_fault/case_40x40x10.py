import numpy as np
from darts.input.input_data import InputData
from darts.tools.keyword_file_tools import load_single_keyword

from case_base import input_data_base

def input_data_case_40x40x10(idata: InputData, case: str):
    input_data_base(idata, case)

    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name
    geom.nx = 40
    geom.ny = 40
    geom.nz = 10

    # idata.gridfile is defined in get_case_files (case_base.py)

    # change properties here

