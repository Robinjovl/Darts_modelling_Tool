import numpy as np
import os
import pandas as pd

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, read_arrays, make_burden_layers, make_full_cube
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
from darts.engines import value_vector

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.gen_cpg_grid import gen_cpg_grid

from darts.models.cicd_model import CICDModel

def get_case_files(case: str):
    prefix = os.path.join('meshes', case)
    gridfile = os.path.join(prefix, 'grid.grdecl')
    propfile = os.path.join(prefix, 'reservoir.in')
    assert os.path.exists(gridfile)
    assert os.path.exists(propfile)
    return gridfile, propfile

def fmt(x):
    return '{:.3}'.format(x)

#####################################################
class Model_CPG(CICDModel):
    def __init__(self, physics_type : str, case : str, grid_out_dir=None, n_points=100):
        super().__init__()
        self.n_points = n_points
        self.physics_type = physics_type
        self.case = case

        # setup filenames
        gridfile, propfile = get_case_files(case)
        self.gridfile = gridfile
        self.propfile = gridfile if propfile == '' else propfile

        bv = 1e6   # boundary volume

        # read grid and props
        arrays = read_arrays(self.gridfile, self.propfile)
        if self.physics_type == 'dead_oil':  # set inactive cells with small porosity (isothermal case)
            arrays['ACTNUM'][arrays['PORO'] < 1e-5] = 0
        elif self.physics_type == 'geothermal':  # process cells with small poro (thermal case)
            arrays['PORO'][arrays['PORO'] < 1e-5] = 1e-5

        if self.physics_type == 'geothermal':
            # add over- and underburden layers
            make_burden_layers(number_of_burden_layers=4, initial_thickness=10, property_dictionary=arrays, burden_layer_prop_value=1e-5)

        self.reservoir = CPG_Reservoir(self.timer, arrays)
        self.reservoir.discretize()

        # add "open" boundaries
        self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
        self.reservoir.apply_volume_depth()

        poro_shale_threshold = 1e-3
        self.reservoir.conduction[np.array(self.reservoir.mesh.poro) <= poro_shale_threshold] = 2.2 * 86.4 # Shale conductivity kJ/m/day/K
        self.reservoir.conduction[np.array(self.reservoir.mesh.poro) > poro_shale_threshold] = 3 * 86.4 # Sandstone conductivity kJ/m/day/K
        self.reservoir.hcap[np.array(self.reservoir.mesh.poro) <= poro_shale_threshold] = 2300 # Shale heat capacity kJ/m3/K
        self.reservoir.hcap[np.array(self.reservoir.mesh.poro) > poro_shale_threshold] = 2450 # Sandstone heat capacity kJ/m3/K

        # add hcap and rcond to be saved into mesh.vtu
        l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
        self.reservoir.global_data.update({'heat_capacity': make_full_cube(self.reservoir.hcap, l2g, g2l),
                                           'rock_conduction': make_full_cube(self.reservoir.conduction, l2g, g2l) })

        self.set_physics()

        # time stepping and convergence parameters
        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=90, runtime=300, tol_newton=1e-2, tol_linear=1e-4)

        self.timer.node["initialization"].stop()

    def set_wells(self):
        # add wells and perforations, 1-based indices
        if self.case == 'brugge':
            i1, j1 = 41, 31  # production well
            i2, j2 = 96, 31  # injection well
        elif self.case == 'case_40x40x10':
            i1, j1 = 10, 20  # production well
            i2, j2 = 30, 20  # injection well
        elif self.case == 'your_case':
            pass

        self.reservoir.add_well('PRD')
        for k in range(1, self.reservoir.nz+1):
            self.reservoir.add_perforation('PRD', cell_index=(i1, j1, k), well_index=None, multi_segment=False,
                                           verbose=True)
        self.reservoir.add_well('INJ')
        for k in range(1, self.reservoir.nz+1):
            self.reservoir.add_perforation('INJ', cell_index=(i2, j2, k), well_index=None, multi_segment=False,
                                           verbose=True)
        print('PRD well:', i1, j1, 'INJ well:', i2, j2)

    def set_initial_pressure_from_file(self, fname : str):
        # set initial pressure
        p_cpp = value_vector()
        load_single_float_keyword(p_cpp, fname, 'PRESSURE', -1)
        p_file = np.array(p_cpp, copy=False)
        p_mesh = np.array(self.reservoir.mesh.pressure, copy=False)
        try:
            actnum = np.array(self.reservoir.actnum, copy=False) # CPG Reservoir
        except:
            actnum = self.reservoir.global_data['actnum']  #Struct reservoir
        p_mesh[:self.reservoir.mesh.n_res_blocks * 2] = p_file[actnum > 0]


    def save_grdecl(self, arrays_save, fname):
        '''
        arr - list of numpy arrays to save, size=nactive
        arr_names - list of array names (keyword)
        '''
        actnum = self.reservoir.global_data['actnum']
        suffix = 'struct'
        if type(self.reservoir) == CPG_Reservoir:
            suffix = 'cpg'
        fname_suf = fname + '_' + suffix + '.grdecl'

        if suffix == 'cpg':
            local_to_global = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
            global_to_local = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)

            save_array(actnum, fname_suf, 'ACTNUM', local_to_global, global_to_local, 'w')
            for arr_name in arrays_save.keys():
                save_array(arrays_save[arr_name], fname_suf, arr_name, local_to_global, global_to_local, 'a')
        else:
            print('save_array is not implemented yet for Struct Reservoir')
            return

    def print_well_rate(self):
        if self.physics_type == 'geothermal':
            # set inj target rate for the next timestep with the production rate value from the previous timestep
            for i, w in enumerate(self.reservoir.wells):
                if self.well_is_inj(w.name):
                    inj_well = w
                else:
                    prod_well = w
            time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
            years = np.array(time_data['time'])[-1]/365.
            pr_col_name = time_data.filter(like=prod_well.name + ' : water rate').columns.to_list()
            pt_col_name = time_data.filter(like=prod_well.name + ' : temperature').columns.to_list()
            ir_col_name = time_data.filter(like=inj_well.name + ' : water rate').columns.to_list()
            rate_prod = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
            temp_prod = np.array(time_data[pt_col_name])[-1][0]  # pick the last timestep value
            rate_inj  = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
            print(fmt(years), 'years:', 'RATE_prod =', fmt(rate_prod), 'RATE_inj =', fmt(rate_inj), 'TEMP_prod =', fmt(temp_prod))
        else:
            # set inj target rate for the next timestep with the production rate value from the previous timestep
            for i, w in enumerate(self.reservoir.wells):
                if self.well_is_inj(w.name):
                    inj_well = w
                else:
                    prod_well = w
            time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
            years = np.array(time_data['time'])[-1] / 365.
            pr_col_name = time_data.filter(like=prod_well.name + ' : oil rate').columns.to_list()
            pp_col_name = time_data.filter(like=prod_well.name + ' : BHP').columns.to_list()
            ir_col_name = time_data.filter(like=inj_well.name + ' : wat rate').columns.to_list()
            ip_col_name = time_data.filter(like=inj_well.name + ' : BHP').columns.to_list()
            rate_prod = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
            bhp_prod = np.array(time_data[pp_col_name])[-1][0]  # pick the last timestep value
            bhp_inj = np.array(time_data[ip_col_name])[-1][0]  # pick the last timestep value
            rate_inj = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
            print(fmt(years), 'years:', 'OIL RATE_prod =', fmt(rate_prod), ' WATER RATE_inj =', fmt(rate_inj), 'BHP_prod =',
                  fmt(bhp_prod), 'BHP_inj =', fmt(bhp_inj))
    def well_is_inj(self, wname : str):  # determine well control by its name
        return "INJ" in wname

    def create_vtk_wells(self, output_directory : str):
        import vtk
        well_vtk_filename = os.path.join(output_directory, 'wells.vtk')
        # Append multiple cylinders into one polydata
        appendFilter = vtk.vtkAppendPolyData()
        def create_tube(center, prolongation=1000):
            # Create points for the polyline
            points = vtk.vtkPoints()
            points.InsertNextPoint(center[0], center[1], center[2] - prolongation)  # Point 1
            points.InsertNextPoint(center[0], center[1], center[2] + prolongation)  # Point 2

            # Create a polyline that connects the points
            lines = vtk.vtkCellArray()
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)  # Number of points
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            lines.InsertNextCell(line)

            # Create a polydata to hold the points and the polyline
            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(lines)

            # Apply vtkTubeFilter to create a tube around the polyline
            tubeFilter = vtk.vtkTubeFilter()
            tubeFilter.SetInputData(polyData)
            tubeFilter.SetRadius(35)  # Tube radius
            tubeFilter.SetNumberOfSides(50)  # Smoothness of the tube
            tubeFilter.Update()

            return tubeFilter.GetOutput()

        for w in self.reservoir.wells:
            for p in w.perforations:
                well_block, res_block_local, well_index, well_indexD = p
                c = self.reservoir.centroids_all_cells[res_block_local].values
                cyl = create_tube(c)
                appendFilter.AddInputData(cyl)
                break  # use only the first perf

        # Update the append filter to combine the polydata
        appendFilter.Update()

        # Write the cylinders to a VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(well_vtk_filename)
        writer.SetInputConnection(appendFilter.GetOutputPort())
        writer.Write()

