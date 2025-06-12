import numpy as np
import os

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, read_arrays, check_arrays, make_burden_layers, make_full_cube
from darts.discretizer import load_single_float_keyword
from darts.engines import value_vector

from darts.tools.gen_cpg_grid import gen_cpg_grid

from darts.models.cicd_model import CICDModel

def fmt(x):
    return '{:.3}'.format(x)

#####################################################

class Model_CPG(CICDModel):
    def __init__(self):
        super().__init__()

    def init_input_arrays(self):
        if self.idata.generate_grid:
            if self.idata.grid_out_dir is None:
                self.idata.gridname = None
                self.idata.propname = None
            else:  # save generated grid to grdecl files
                os.makedirs(self.idata.grid_out_dir, exist_ok=True)
                self.idata.gridname = os.path.join(self.idata.grid_out_dir, 'grid.grdecl')
                self.idata.propname = os.path.join(self.idata.grid_out_dir, 'reservoir.in')
            arrays = gen_cpg_grid(nx=self.idata.geom.nx, ny=self.idata.geom.ny, nz=self.idata.geom.nz,
                                  dx=self.idata.geom.dx, dy=self.idata.geom.dy, dz=self.idata.geom.dz,
                                  start_z=self.idata.geom.start_z,
                                  permx=self.idata.rock.permx, permy=self.idata.rock.permy, permz=self.idata.rock.permz,
                                  poro=self.idata.rock.poro,
                                  gridname=self.idata.gridname, propname=self.idata.propname)
        else:
            # read grid and rock properties
            arrays = read_arrays(self.idata.gridfile, self.idata.propfile)
        return arrays

    def init_reservoir(self, arrays):
        check_arrays(arrays)
        if not self.physics.thermal:  # set inactive cells with small porosity (isothermal case)
            arrays['ACTNUM'][arrays['PORO'] < self.idata.geom.min_poro] = 0
        else:  # process cells with small poro (thermal case)
            arrays['PORO'][arrays['PORO'] < self.idata.geom.min_poro] = self.idata.geom.min_poro
            # allow small flow since there might pressure change appear due to the temperature change
            for arr in ['PERMX', 'PERMY', 'PERMZ']:
                arrays[arr][arrays[arr] < self.idata.geom.min_perm] = self.idata.geom.min_perm

        if self.idata.geom.burden_layers > 0:
            # add over- and underburden layers
            make_burden_layers(number_of_burden_layers=self.idata.geom.burden_layers,
                               initial_thickness=self.idata.geom.burden_init_thickness,
                               property_dictionary=arrays,
                               burden_layer_prop_value=self.idata.rock.burden_prop)

        self.reservoir = CPG_Reservoir(self.timer, arrays, minpv=self.idata.geom.minpv, faultfile=self.idata.geom.faultfile)
        # discretize right away to be able to modify the boundary volume
        self.reservoir.discretize()

        # store modified arrrays (with burden layers) for output to grdecl
        self.reservoir.input_arrays = arrays

        volume = np.array(self.reservoir.mesh.volume, copy=False)
        poro = np.array(self.reservoir.mesh.poro, copy=False)
        print("Pore volume = " + str(sum(volume[:self.reservoir.mesh.n_blocks] * poro)))

        # imitate open-boundaries with a large volume
        bv = self.idata.geom.bound_volume   # volume, will be assigned to each boundary cell [m3]
        self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
        self.reservoir.apply_volume_depth()

        poro_shale_threshold = self.idata.rock.poro_shale_threshold  # short name
        poro = np.array(self.reservoir.mesh.poro)
        self.reservoir.conduction[poro <= poro_shale_threshold] = self.idata.rock.conduction_shale
        self.reservoir.conduction[poro > poro_shale_threshold] = self.idata.rock.conduction_sand
        self.reservoir.hcap[poro <= poro_shale_threshold] = self.idata.rock.hcap_shale
        self.reservoir.hcap[poro > poro_shale_threshold] = self.idata.rock.hcap_sand

        # add hcap and rcond to be saved into mesh.vtu
        l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
        self.reservoir.global_data.update({'heat_capacity': make_full_cube(self.reservoir.hcap.copy(), l2g, g2l),
                                           'rock_conduction': make_full_cube(self.reservoir.conduction.copy(), l2g, g2l) })

    def set_wells(self):
        # add wells and perforations, 1-based IJK indices
        if hasattr(self.idata, 'schfile'):
            # apply to the reservoir from idata filled before by idata.read_and_add_perforations()
            for wname, wdata in self.idata.well_data.wells.items():
                self.reservoir.add_well(wname)
                for perf_tuple in wdata.perforations:
                    perf = perf_tuple[1]
                    # adjust to account for added overburden layers
                    perf_ijk_new = (perf.loc_ijk[0], perf.loc_ijk[1], perf.loc_ijk[2] + self.idata.geom.burden_layers)
                    # take well index if it was defined in sch file, otherwise take the default one from idata
                    wi = perf.well_index if perf.well_index is not None else self.idata.geom.well_index
                    self.reservoir.add_perforation(wname,
                                                   cell_index=perf_ijk_new,
                                                   well_index=wi, well_indexD=self.idata.geom.well_indexD,
                                                   multi_segment=perf.multi_segment, verbose=True)
        else:
            # add wells and perforations, 1-based indices
            for wname, wdata in self.idata.well_data.wells.items():
                self.reservoir.add_well(wname)
                for k in range(1 + self.idata.geom.burden_layers,  self.reservoir.nz+1-self.idata.geom.burden_layers):
                    self.reservoir.add_perforation(wname,
                                                   cell_index=(wdata.location.I, wdata.location.J, k),
                                                   well_index=self.idata.geom.well_index, well_indexD=self.idata.geom.well_indexD,
                                                   multi_segment=False, verbose=True)

    def well_is_inj(self, wname : str):  # determine well control by its name
        return "INJ" in wname

    def do_after_step(self):
        # save to grdecl file after each time step
        # self.reservoir.save_grdecl(self.get_arrays(), os.path.join(out_dir, 'res_' + str(ti+1)))
        self.physics.engine.report()
        self.print_well_rate()

    def set_well_controls(self):  # dummy. just to pass through model.init()
        self.set_well_controls_idata()


