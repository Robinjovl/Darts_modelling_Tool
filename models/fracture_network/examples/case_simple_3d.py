# -*- coding: utf-8 -*-
import os
from .input_default import input_data_default

def input_data_case_3D_base():
    '''
    Fills input data for case 3D input with default data, which is then complemented (for instance, mesh)
    and can be overwritten (for instance, well locations) in specific cases.
    This helps to avoid duplication.
    '''
    idata = input_data_default()
    input_data = idata.geom  # a short name

    idata.geom['mesh_type'] = '3D'
    input_data['bondary_volume_xy'] = 0 # the code for it is slow for 3D
    idata.geom['matrix_tags'] = [0]
    idata.geom['frac_tag_start'] = 7
    input_data['frac_geom_type'] = 'triangle'

    z_middle = input_data['z_top'] + 0.5 * input_data['height_res']  # perforation depth [m]

    # well locations (multiple perforations - through all depth of the reservoir)
    idata.geom['well_coords'] = dict()
    # inj well
    input_data['well_coords']['I1'] = [250., 750., input_data['z_top'], input_data['z_top'] + input_data['height_res']]  # X, Y, Z1, Z2
    # prod well
    input_data['well_coords']['P1'] = [250., 750., input_data['z_top'], input_data['z_top'] + input_data['height_res']]  # X, Y, Z1, Z2

    # initial pressure and temperature at reservoir depth for well controls
    idata.initial.type ='gradient'
    idata.initial.initial_pressure = idata.initial.pressure_at_ref_depth + idata.initial.pressure_gradient * (z_middle - idata.initial.reference_depth_for_pressure) / 1000.
    idata.initial.initial_temperature = idata.initial.temperature_at_ref_depth + idata.initial.temperature_gradient * (z_middle - idata.initial.reference_depth_for_temperature) / 1000.

    wctrl = idata.well_data.controls  #short name
    wctrl.prod_rate = 10  # m3/day. if None, well will work under BHP control
    wctrl.inj_rate = 10   # m3/day. if None, well will work under BHP control

    return idata

def input_data_case_3D_strike0_dip90():
    '''
        This cases uses a hexahedral rectangular mesh, the fracture is located in XY-plane
        for testing purposes with different cell shapes, fracture shape is 'quad' in this case
    '''
    idata = input_data_case_3D_base()
    input_data = idata.geom  # a short name

    input_data['case_name'] = '3D_strike0_dip0'
    input_data['mesh_filename'] = os.path.join('meshes_3D', 'dip90.msh')
    input_data['frac_geom_type'] = 'quad'

    return idata


def input_data_case_3D_strike0_dip45():
    '''
        This cases uses a hexahedral mesh, the fracture dip angle is 45 degrees
        For testing purposes with different cell shapes, fracture shape is 'quad' in this case
    '''
    idata = input_data_case_3D_base()
    input_data = idata.geom  # a short name

    input_data['case_name'] = '3D_strike0_dip45'
    input_data['mesh_filename'] = os.path.join('meshes_3D', 'dip45.msh')
    input_data['frac_geom_type'] = 'quad'

    return idata

def input_data_case_3D_strike0_dip0():
    '''
        This cases uses a tetrahedral mesh, the fracture is locates in XZ-plane
        For testing purposes with different cell shapes, fracture shape is 'tetra' in this case
    '''
    idata = input_data_case_3D_base()
    input_data = idata.geom  # a short name

    input_data['case_name'] = '3D_strike0_dip0'
    input_data['mesh_filename'] = os.path.join('meshes_3D', 'mesh_strike0_dip0.msh')

    return idata

def input_data_case_3D_no_fractures():
    '''
        This cases uses a tetrahedral mesh, there are no fractures
    '''
    idata = input_data_case_3D_base()
    input_data = idata.geom  # a short name

    input_data['case_name'] = '3D_no_fractures'
    input_data['mesh_filename'] = os.path.join('meshes_3D', 'dipNone.msh')

    # well locations
    idata.geom['well_coords'] = dict()
    # multiple perforations - through all depth of the reservoir
    input_data['well_coords']['I1'] = [200., 500., 2000., 2500.]  # X, Y, Z1, Z2
    input_data['well_coords']['P1'] = [800., 500., 2000., 2500.]  # X, Y, Z1, Z2

    return idata
