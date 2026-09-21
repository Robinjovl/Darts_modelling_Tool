import numpy as np
import os
from darts.input.input_data import InputData
from case_base import input_data_base, get_case_files

def input_data_case_40x40x10(idata: InputData, case: str):
    input_data_base(idata, case)

    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name

    # idata.gridfile is defined in get_case_files (case_base.py)

    # change properties here

def input_data_case_40x40x10_hcap(idata: InputData, case: str):
    input_data_case_40x40x10(idata, case)

    idata.gridfile = 'grid.grdecl'
    idata.schfile = 'sch.inc'
    # rock heat capacity is defined in the input file
    idata.propfile = 'reservoir_hcap.in'
    idata.gridfile, idata.propfile, idata.schfile = get_case_files(case, idata.gridfile, idata.propfile, idata.schfile)

def input_data_case_40x40x10_regions(idata: InputData, case: str):
    input_data_case_40x40x10(idata, case)

    idata.gridfile = 'grid.grdecl'
    idata.schfile = 'sch.inc'
    # geological unit array (ROCKNUM): 1, 2, 3, 4 is defined in this input file
    idata.propfile = 'reservoir_regions.in'
    idata.gridfile, idata.propfile, idata.schfile = get_case_files(case, idata.gridfile, idata.propfile, idata.schfile)

    # rock properties are defined for each geological unit
    idata.poro_geo_units = [0.01, 0.2, 0.1, 0.15]
    idata.perm_geo_units = [0.01, 0.2, 0.1, 0.15]  # mD
    idata.rcond_geo_units = [190, 280, 210, 250]  # kJ/m/day/K
    idata.hcap_geo_units = [2100, 2400, 2200, 2350]  # kJ/m3/K

