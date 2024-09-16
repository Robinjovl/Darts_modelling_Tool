import numpy as np
import os
import pandas as pd

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, read_arrays, make_burden_layers
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
    sch_file = os.path.join(prefix, 'SCH.INC')
    assert os.path.exists(gridfile)
    assert os.path.exists(propfile)
    assert os.path.exists(sch_file)
    return gridfile, propfile, sch_file

def fmt(x):
    return '{:.3}'.format(x)

#####################################################
class Model_CPG(CICDModel):
    def __init__(self, physics_type='geothermal', discr_type='cpp', case='generate', grid_out_dir=None, n_points=100):
        super().__init__()
        self.n_points = n_points
        self.physics_type = physics_type
        self.discr_type = discr_type
        self.case = case
        self.generate_grid = 'generate' in case

        if self.generate_grid:
            if case == 'generate_51x51x1':
                self.nx = 51
                self.ny = 51
                self.nz = 1
                self.dx = 4000. / self.nx
                self.dy = self.dx
                self.dz = 100. / self.nz
                self.start_z = 2000  # top reservoir depth
            elif case == 'generate_5x3x4':
                self.nx = 5
                self.ny = 3
                self.nz = 4
                self.start_z = 1000  # top reservoir depth
                # non-uniform layers thickness
                self.dx = np.array([500, 200, 100, 300, 500])
                self.dy = np.array([1000, 700, 300])
                self.dz = np.array([100, 150, 180, 120])
            poro = 0.2
            permx = 100
            permy = 100
            permz = 10
        else:  # read from files
            gridfile, propfile, sch_fname = get_case_files(case)
            self.gridfile = gridfile
            self.propfile = gridfile if propfile == '' else propfile
            self.sch_fname = sch_fname

        hcap = 2200     # heat capacity [kJ/m3/K]
        rcond = 181.44  # heat conduction [kJ/m/day/K]

        bv = 1e6   # boundary volume

        if discr_type == 'cpg':
            if self.generate_grid:
                if grid_out_dir is None:
                    gridname = None
                    propname = None
                else:  # save generated grid to grdecl files
                    os.makedirs(grid_out_dir, exist_ok=True)
                    gridname = os.path.join(grid_out_dir, 'grid.grdecl')
                    propname = os.path.join(grid_out_dir, 'reservoir.in')
                arrays = gen_cpg_grid(nx=self.nx, ny=self.ny, nz=self.nz,
                                      dx=self.dx, dy=self.dy, dz=self.dz, start_z=self.start_z,
                                      permx=permx, permy=permy, permz=permz, poro=poro,
                                      gridname=gridname, propname=propname)
            else:
                arrays = read_arrays(self.gridfile, self.propfile)
                if self.physics_type == 'dead_oil':  # set inactive cells with small porosity (isothermal case)
                    arrays['ACTNUM'][arrays['PORO'] < 1e-5] = 0
                elif self.physics_type == 'geothermal':  # process cells with small poro (thermal case)
                    arrays['PORO'][arrays['PORO'] < 1e-5] = 1e-5

                if self.physics_type == 'geothermal':
                    # add over- and underburden layers
                    make_burden_layers(number_of_burden_layers=4, initial_thickness=10, property_dictionary=arrays)

            self.reservoir = CPG_Reservoir(self.timer, arrays)
            self.reservoir.discretize()
            # add "open" boundaries
            self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
            self.reservoir.apply_volume_depth()

            cond_mesh = np.array(self.reservoir.mesh.rock_cond, copy=False)
            hcap_mesh = np.array(self.reservoir.mesh.heat_capacity, copy=False)
            cond_mesh[np.array(self.reservoir.mesh.poro) <= 1e-3] = 2.2 * 86.4 # Shale conductivity kJ/m/day/K
            cond_mesh[np.array(self.reservoir.mesh.poro) > 1e-3] = 3 * 86.4 # Sandstone conductivity kJ/m/day/K
            hcap_mesh[np.array(self.reservoir.mesh.poro) <= 1e-3] = 2300 # Shale heat capacity kJ/m3/K
            hcap_mesh[np.array(self.reservoir.mesh.poro) > 1e-3] = 2450 # Sandstone heat capacity kJ/m3/K

        elif discr_type == 'struct':
            if self.generate_grid:
                self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz,
                                                 dx=self.dx, dy=self.dy, dz=self.dz, start_z=self.start_z, depth=None,
                                                 permx=permx, permy=permy, permz=permz, poro=poro,
                                                 hcap=hcap, rcond=rcond)
            else:
                self.init_struct_reservoir(hcap=hcap, rcond=rcond)
            self.reservoir.boundary_volumes['yz_minus'] = bv
            self.reservoir.boundary_volumes['yz_plus'] = bv
            self.reservoir.boundary_volumes['xz_minus'] = bv
            self.reservoir.boundary_volumes['xz_plus'] = bv

        self.set_physics()

        # time stepping and convergence parameters
        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=90, runtime=300, tol_newton=1e-2, tol_linear=1e-4)

        self.timer.node["initialization"].stop()

    def init_struct_reservoir(self, hcap, rcond):
        '''
        initialize StructReservoir by the data in files self.gridfile and self.propfile
        '''
        dims_cpp = index_vector_discr()
        load_single_int_keyword(dims_cpp, self.gridfile, "SPECGRID", 3)
        dims = np.array(dims_cpp, copy=False)

        permx_cpp, permy_cpp, permz_cpp = value_vector_discr(), value_vector_discr(), value_vector_discr()
        load_single_float_keyword(permx_cpp, self.propfile, 'PERMX', -1)
        load_single_float_keyword(permy_cpp, self.propfile, 'PERMY', -1)
        permx = np.array(permx_cpp, copy=False)
        permy = np.array(permy_cpp, copy=False)
        for perm_str in ['PERMEABILITYXY', 'PERMEABILITY']:
            if permx.size == 0 or permy.size == 0:
                load_single_float_keyword(permx_cpp, self.propfile, perm_str, -1)
                permy_cpp = permx_cpp
                permx = np.array(permx_cpp, copy=False)
                permy = np.array(permy_cpp, copy=False)
        load_single_float_keyword(permz_cpp, self.propfile, 'PERMZ', -1)
        permz = np.array(permz_cpp, copy=False)

        poro_cpp = value_vector_discr()
        load_single_float_keyword(poro_cpp, self.propfile, 'PORO', -1)
        poro = np.array(poro_cpp, copy=False)

        coord_cpp = value_vector_discr()
        load_single_float_keyword(coord_cpp, self.gridfile, 'COORD', -1)
        coord = np.array(coord_cpp, copy=False)

        zcorn_cpp = value_vector_discr()
        load_single_float_keyword(zcorn_cpp, self.gridfile, 'ZCORN', -1)
        zcorn = np.array(zcorn_cpp, copy=False)

        actnum_cpp = index_vector_discr()
        actnum = np.array([])
        for fname in [self.gridfile, self.propfile]:
            if actnum.size == 0:
                load_single_int_keyword(actnum_cpp, fname, 'ACTNUM', -1)
                actnum = np.array(actnum_cpp, copy=False)
        if actnum.size == 0:
            actnum = np.ones(dims[0] * dims[1] * dims[2])
            print('No ACTNUM found in input files. ACTNUM=1 will be used')

        if self.physics_type == 'dead_oil':  # make cells with zero porosity (make sense if not thermal)
            actnum[poro == 0.0] = 0
        elif self.physics_type == 'geothermal':  # makes sense for thermal
            poro[poro == 0.0] = 1.E-4

        self.reservoir = StructReservoir(self.timer, nx=dims[0], ny=dims[1], nz=dims[2], dx=0, dy=0, dz=0,
                                         permx=permx, permy=permy, permz=permz, poro=poro, hcap=hcap, rcond=rcond,
                                         depth=None, actnum=actnum, coord=coord, zcorn=zcorn, is_cpg=True)
        return

    def set_wells(self):
        # add wells
        if not self.generate_grid:
            self.read_and_add_perforations(self.reservoir, sch_fname=self.sch_fname, verbose=True)
        else:
            if self.case == 'generate_51x51x1':
                i1, j1 = self.nx//2 - int(500//self.dx), self.ny//2  # I = 0.5 km to the left from the center
                i2, j2 = self.nx//2 + int(500//self.dx), self.ny//2  # I = 0.5 km to the right from the center
            elif self.case == 'generate_5x3x4':
                i1, j1 = 1, 1
                i2, j2 = 5, 3

            self.reservoir.add_well('PRD')
            for k in range(1, self.reservoir.nz+1):
                self.reservoir.add_perforation('PRD', cell_index=(i1, j1, k), well_index=None, multi_segment=False,
                                               verbose=True)
            self.reservoir.add_well('INJ')
            for k in range(1, self.reservoir.nz+1):
                self.reservoir.add_perforation('INJ', cell_index=(i2, j2, k), well_index=None, multi_segment=False,
                                               verbose=True)
            print('DX', self.dx, 'DY', self.dy, 'PRD:', i1, j1, 'INJ:', i2, j2)

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

    def read_and_add_perforations(self, reservoir, sch_fname, well_index: float = None, verbose: bool = False):
        '''
        read COMPDAT from SCH file in Eclipse format, add wells and perforations
        note: uses only I,J,K1,K2 parameters from COMPDAT
        '''
        if sch_fname is None:
            return
        print('reading wells (COMPDAT) from', sch_fname)
        well_dia = 0.152
        well_rad = well_dia / 2

        keep_reading = True
        prev_well_name = ''
        with open(sch_fname) as f:
            while keep_reading:
                buff = f.readline()
                if 'COMPDAT' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            CompDat = buff.split()
                            wname = CompDat[0].strip('"').strip("'") #remove quotas (" and ')
                            if len(CompDat) != 0 and '/' != wname:  # skip the empty line and '/' line
                                # define well
                                if wname == prev_well_name:
                                    pass
                                else:
                                    reservoir.add_well(wname)
                                    prev_well_name = wname
                                # define perforation
                                i1 = int(CompDat[1])
                                j1 = int(CompDat[2])
                                k1 = int(CompDat[3])
                                k2 = int(CompDat[4])

                                for k in range(k1, k2 + 1):
                                    reservoir.add_perforation(wname, cell_index=(i1, j1, k), well_radius=well_rad,
                                                              well_index=well_index, multi_segment=False, verbose=verbose)

                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break
        print('WELLS read from SCH file:', len(reservoir.wells))

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

