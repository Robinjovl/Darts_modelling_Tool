from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import DartsModel
from darts.engines import sim_params, well_control_iface
from darts.engines import (
    conn_mesh,
    index_vector,
    ms_well,
    ms_well_vector,
    timer_node,
    value_vector,
)
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.reservoirs.reservoir_base import ReservoirBase


class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()

        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

    def build_lgr_definition(self):
        """
        -parent name
        -i,j,k range in parent grid (1-based)
        -local name
        """
        lgrs = {}
        # Define lgr0
        parent_grid_name = 'global'
        lgr_coords_in_parent_grid = {
        "i_range": [2, 2],
        "j_range": [2, 2],
        "k_range": [1, 1],
        "refine": [3, 3, 1],
        "tag" : "inj"
        }

        lgrs['lgr0'] = {
            'parent_grid_name': parent_grid_name,
            'lgr_coords_in_parent_grid': lgr_coords_in_parent_grid,
        }
        # Define lgr1
        parent_grid_name = 'global'
        lgr_coords_in_parent_grid = {
        "i_range": [4, 4],
        "j_range": [2, 2],
        "k_range": [1, 1],
        "refine": [3, 3, 1],
        "tag" : "prd"
        }
        lgrs['lgr1'] = {
            'parent_grid_name': parent_grid_name,
            'lgr_coords_in_parent_grid': lgr_coords_in_parent_grid,
        }
        return lgrs

    def convert_ijk_to_gindex_0based(self, i_1based: int, j_1based: int, k_1based:int, nx:int, ny:int) -> int:
        """
        Convert (i,j,k) in 1-based to global index in 0-based for nz=1 case
        """
        g_index_0based = (k_1based - 1) * nx * ny + (j_1based - 1) * nx + (i_1based - 1)
        return g_index_0based

    def create_actnum_with_lgr(self, nx0, ny0, refined_ij_list):
        actnum0 = np.ones(nx0 * ny0, dtype=np.int32)
        for i_c,j_c in refined_ij_list:
            g = self.convert_ijk_to_gindex_0based(i_c, j_c, 1, nx0, ny0)
            actnum0[g] = 0  # deactivate the coarse cell that will be refined
        return actnum0

    def auto_image_grid_dx_dy(self, dx_parent, dy_parent, rx, ry):
        dx_f = dx_parent / rx
        dy_f = dy_parent / ry
        dx_image = np.array([dx_parent, dx_f,dx_f,dx_f, dx_parent], dtype=float)
        dy_image = np.array([dy_parent, dy_f,dy_f,dy_f, dy_parent], dtype=float)
        return dx_image, dy_image

    def set_reservoir(self):
        self.lgrs = self.build_lgr_definition()

        refined_cells_ij = []
        for name, cfg in self.lgrs.items():
            assert cfg['parent_grid_name'] == 'global'
            i1, i2 = cfg['lgr_coords_in_parent_grid']['i_range']
            j1, j2 = cfg['lgr_coords_in_parent_grid']['j_range']
            k1, k2 = cfg['lgr_coords_in_parent_grid']['k_range']
            refined_cells_ij.append( (i1, j1) )  # only nz=1 case

        self.refined_cells_ij = refined_cells_ij

        #Build Level0 with actnum
        nx0, ny0, nz0 = 5, 3, 1  # global grid size
        dx0, dy0, dz0 = 100, 100, 5
        permx0, permy0, permz0 = 50, 50, 50
        poro0 = 0.1
        depth0 = 2000

        actnum0 = self.create_actnum_with_lgr(nx0, ny0, refined_cells_ij)

        self.level0 = StructReservoir(self.timer, nx=nx0, ny=ny0, nz=nz0, dx=dx0, dy=dy0, dz=dz0,
                                         permx=permx0, permy=permy0, permz=permz0, poro=poro0, depth=depth0,actnum=actnum0)


        # Build Level1 grids and imaginary grids
        self.level1 = {}
        self.level1_imag = {}
        for name, cfg in self.lgrs.items():
            rx, ry, rz = cfg['lgr_coords_in_parent_grid']['refine']

            nx1,ny1,nz1 = rx, ry, rz
            dx1 = dx0 / rx
            dy1 = dy0 / ry
            dz1 = dz0 / rz

            self.level1[name] = StructReservoir(self.timer, nx=nx1, ny=ny1, nz=nz1, dx=dx1, dy=dy1, dz=dz1,
                                        permx=permx0, permy=permy0, permz=permz0, poro=poro0, depth=depth0)

            dx_imag, dy_imag = self.auto_image_grid_dx_dy(dx0, dy0, rx, ry)
            self.level1_imag[name] = StructReservoir(self.timer, nx=5, ny=5, nz=1, dx=dx_imag, dy=dy_imag, dz=dz0,
                                        permx=permx0, permy=permy0, permz=permz0, poro=poro0, depth=depth0)



        cm_all, cp_all, T_all, T_all_therm, meta = self.assemble_lgr_connections_eclipse()

       # assemble properties
        self.level0.discretize()
        disc0 = self.level0.discretizer
        l2g0 = disc0.local_to_global          # l2g0 has already eliminated inactive cells

        dx0_arr = disc0.convert_to_flat_array(self.level0.global_data['dx'], 'dx')[l2g0]
        dy0_arr = disc0.convert_to_flat_array(self.level0.global_data['dy'], 'dy')[l2g0]
        dz0_arr = disc0.convert_to_flat_array(self.level0.global_data['dz'], 'dz')[l2g0]
        kx0_arr = disc0.convert_to_flat_array(self.level0.global_data['permx'], 'permx')[l2g0]
        ky0_arr = disc0.convert_to_flat_array(self.level0.global_data['permy'], 'permy')[l2g0]
        kz0_arr = disc0.convert_to_flat_array(self.level0.global_data['permz'], 'permz')[l2g0]
        poro0_arr  = disc0.convert_to_flat_array(self.level0.global_data['poro'],  'poro')[l2g0]
        depth0_arr = disc0.convert_to_flat_array(self.level0.global_data['depth'], 'depth')[l2g0]
        vol_g0 = np.ones(self.level0.nx*self.level0.ny*self.level0.nz, dtype=float) * (100*100*5)
        volume0_arr = vol_g0[l2g0]

        dx_list = [dx0_arr]; dy_list = [dy0_arr]; dz_list = [dz0_arr]
        kx_list = [kx0_arr]; ky_list = [ky0_arr]; kz_list = [kz0_arr]
        poro_list = [poro0_arr]; depth_list = [depth0_arr]; volume_list = [volume0_arr]

        for name in meta['lgr_orders']:
            self.level1[name].discretize()
            disc1 = self.level1[name].discretizer

            dx_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["dx"], "dx"))
            dy_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["dy"], "dy"))
            dz_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["dz"], "dz"))
            kx_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permx"], "permx"))
            ky_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permy"], "permy"))
            kz_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["permz"], "permz"))

            poro_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["poro"], "poro"))
            depth_list.append(disc1.convert_to_flat_array(self.level1[name].global_data["depth"], "depth"))
            volume_list.append(np.ones(self.level1[name].n, dtype=float) * (dx0/3) * (dy0/3) * dz0)

        dx = np.concatenate(dx_list)
        dy = np.concatenate(dy_list)
        dz = np.concatenate(dz_list)
        kx = np.concatenate(kx_list)
        ky = np.concatenate(ky_list)
        kz = np.concatenate(kz_list)
        poro = np.concatenate(poro_list)
        depth = np.concatenate(depth_list)
        volume = np.concatenate(volume_list)
        rcon = np.zeros_like(poro)
        hcap = np.zeros_like(poro)



        self.reservoir = LGRReservoir(self.timer,
                                        cell_m=cm_all,
                                        cell_p=cp_all,
                                        tran=T_all,
                                        tran_thermal=T_all_therm,
                                        poro=poro,
                                        rcond=rcon,
                                        hcap=hcap,
                                        depth=depth,
                                        volume=volume,
                                        dx=dx,
                                        dy=dy,
                                        dz=dz,
                                        kx=kx,
                                        ky=ky,
                                        kz=kz,
                                        )
        self.lgr_meta = meta
        return


    def assemble_lgr_connections_eclipse(self):
        # STEP 1 discretize Level 0
        self.level0.discretize()
        disc0 = self.level0.discretizer
        g2l0 = disc0.global_to_local
        l2g0 = disc0.local_to_global
        n0_act = len(l2g0)

        cm0 = self.level0.cell_m
        cp0 = self.level0.cell_p
        T0 = self.level0.tran
        T0_therm = self.level0.tran_thermal


        # step 2 discretize each fine LGR and build global offsets
        lgr_orders = list(self.lgrs.keys())
        lgr_offsets = {} # name -> start global index in assembled system

        fine_conn_parts = []
        fine_T_parts = []
        fine_Tt_parts = []

        offset = n0_act
        for name in lgr_orders:
            self.level1[name].discretize()
            disc1 = self.level1[name].discretizer
            cm1, cp1, T1, T1_therm = disc1.calc_structured_discr()
            lgr_offsets[name] = offset
            fine_conn_parts.append((cm1 + offset, cp1 + offset))
            fine_T_parts.append(T1)
            fine_Tt_parts.append(T1_therm)
            offset += self.level1[name].n

        # step3 fine-coarse connections via imaginary grids
        fc_cm = []
        fc_cp = []
        fc_T = []
        fc_Tt = []
        nxim, nyim = 5, 5  # imaginary grid size is 5x5
        fine_im = []
        for j in range(1,4):
            for i in range(1,4):
                fine_im.append(j*nxim + i)
        fine_im_set = set(fine_im)
        coarse_im = []
        for i in range(1,4):
            coarse_im.append(i) # top edge
            coarse_im.append(4*nxim + i) # bottom edge
        for j in range(1,4):
            coarse_im.append(j*nxim) # left edge
            coarse_im.append(j*nxim + 4) # right edge
        coarse_im_set = set(coarse_im)

        # mapping: imaginary fine index (5*5) -> local fine index (3*3)
        imag_fine_to_local = {}
        k = 0
        for j in range(1,4):
            for i in range(1,4):
                imag_idx = j*nxim + i # index of 5*5 grid
                imag_fine_to_local[imag_idx] = k # index of 3*3 grid
                k += 1
        # the imaginary index of those cells adjacent to fine cells
        left_im = [j * nxim for j in range(1,4)]
        right_im = [4 + j*nxim for j in range(1,4)]
        up_im = [i for i in range(1,4)]
        down_im = [4*nxim + i for i in range(1,4)]

        for name in lgr_orders:
            cfg = self.lgrs[name]
            i = cfg['lgr_coords_in_parent_grid']['i_range'][0]
            j = cfg['lgr_coords_in_parent_grid']['j_range'][0]
            rx,ry,rz = cfg['lgr_coords_in_parent_grid']['refine']

            self.level1_imag[name].discretize()
            disc_im = self.level1_imag[name].discretizer
            cmi, cpi, Ti, Ti_therm = disc_im.calc_structured_discr()

            is_fine_coarse = np.array([(cm in fine_im_set and cp in coarse_im_set)
                                     or (cp in fine_im_set and cm in coarse_im_set)
                                     for cm, cp in zip(cmi, cpi)], dtype= bool)
            # boundary connection (fine-coarse)
            cmi_fc = cmi[is_fine_coarse]
            cpi_fc = cpi[is_fine_coarse]
            Ti_fc = Ti[is_fine_coarse]
            Ti_therm_fc = Ti_therm[is_fine_coarse]

            # Map the fine index of imaginary to the global index of level 1
            # map the coarse index to the real index of level 0
            nx0 = self.level0.nx
            # compute neigbor indices in level0 after eliminating inactive cells
            nbr_g = {
                "left" : self.convert_ijk_to_gindex_0based(i-1, j, 1, nx0, self.level0.ny),
                "right" : self.convert_ijk_to_gindex_0based(i+1, j, 1, nx0, self.level0.ny),
                "up" : self.convert_ijk_to_gindex_0based(i, j-1, 1, nx0, self.level0.ny),
                "down" : self.convert_ijk_to_gindex_0based(i, j+1, 1, nx0, self.level0.ny),
            }
            nbr_l = {k: int(g2l0[v]) for k, v in nbr_g.items()} # local indices in level0 after actnum squeeze

            # map imag coarse ring to lvel0 local index
            image_coarse_to_level0 = {}
            for idx in left_im:
                image_coarse_to_level0[idx] = nbr_l['left']
            for idx in right_im:
                image_coarse_to_level0[idx] = nbr_l['right']
            for idx in up_im:
                image_coarse_to_level0[idx] = nbr_l['up']
            for idx in down_im:
                image_coarse_to_level0[idx] = nbr_l['down']

            fine_global_offset = lgr_offsets[name]

            for cm, cp, t, tt in zip(cmi_fc,cpi_fc, Ti_fc, Ti_therm_fc):
                # imag_fine_to_local is a dict: imag index -> local fine index (3*3) of level1
                # image_coarse_to_level0 is a dict: imag index -> local coarse index of level0 (after actnum squeeze)
                if cm in imag_fine_to_local and cp in image_coarse_to_level0:
                    fine_local = imag_fine_to_local[cm]
                    coarse_local = image_coarse_to_level0[cp]
                elif cp in imag_fine_to_local and cm in image_coarse_to_level0:
                    fine_local = imag_fine_to_local[cp]
                    coarse_local = image_coarse_to_level0[cm]
                else:
                    continue
                fine_global = fine_local + fine_global_offset # global index of fine cell
                fc_cm.append(coarse_local)
                fc_cp.append(fine_global)
                fc_T.append(t)
                fc_Tt.append(tt)

        # assemble all connections

        ## coarse-coarse connections
        cm_parts = [cm0]
        cp_parts = [cp0]
        T_parts = [T0]
        Tt_parts = [T0_therm]
        ## fine-fine connections
        for (cmg, cpg), t, tt in zip(fine_conn_parts, fine_T_parts, fine_Tt_parts):
            cm_parts.append(cmg)
            cp_parts.append(cpg)
            T_parts.append(t)
            Tt_parts.append(tt)
        ## fine-coarse connections
        cm_parts.append(np.asarray(fc_cm, dtype= int))
        cp_parts.append(np.asarray(fc_cp, dtype= int))
        T_parts.append(np.asarray(fc_T, dtype= float))
        Tt_parts.append(np.asarray(fc_Tt, dtype= float))

        cm_all = np.concatenate(cm_parts)
        cp_all = np.concatenate(cp_parts)
        T_all = np.concatenate(T_parts)
        T_all_therm = np.concatenate(Tt_parts)

        print(f'cm is {cm_all}')
        print(f'cp is {cp_all}')

        meta = {
            "lgr_orders": lgr_orders,
            "lgr_offsets": lgr_offsets,
            "n0_act": n0_act,
            "well_local_center": 4,
        }
        return cm_all, cp_all, T_all, T_all_therm, meta


    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_well("P1")

        center_local = self.lgr_meta['well_local_center']

        inj_lgr = "lgr0"
        prd_lgr = "lgr1"
        inj_global = self.lgr_meta["lgr_offsets"][inj_lgr] + center_local
        prd_global = self.lgr_meta["lgr_offsets"][prd_lgr] + center_local

        self.reservoir.add_perforation("I1", cell_index=inj_global)

        self.reservoir.add_perforation("P1", cell_index=prd_global)

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        components = ['CO2', 'H2O']
        phases = ['CO2_rich', 'aqueous']
        Mw = [44.01, 18.015]

        temperature = 300
        property_container = PropertyContainer(phases_name=phases, components_name=components,
                                               Mw=Mw, min_z=self.zero / 10, temperature=temperature)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(components), [4, 1e-1], self.zero)
        property_container.density_ev = dict([('CO2_rich', DensityBasic(compr=1e-3, dens0=200)),
                                              ('aqueous', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('CO2_rich', ConstFunc(0.05)),
                                                ('aqueous', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('CO2_rich', PhaseRelPerm("gas")),
                                               ('aqueous', PhaseRelPerm("oil"))])

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=200, min_p=1, max_p=300, min_z=self.zero/10, max_z=1-self.zero/10)
        # property_container.output_props = {
        #     "sat0": lambda: property_container.sat[0],
        #     "dens0": lambda: property_container.dens[0],
        #     "nu0": lambda: property_container.nu[0],
        #     "x00": lambda: property_container.x[0,0]
        #     }

        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 50,
                              self.physics.vars[1]: self.zero,
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        inj_composition = [1.0 - self.zero]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=140., inj_composition=inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=50.)

class LGRReservoir(ReservoirBase):
    """
    A simple reservoir wrapper that takes pre-assembled
    connection lists, Transmissibility and cell properties
    , then build a new conn_mesh
    """
    def __init__(self, timer, cell_m, cell_p, tran, tran_thermal,
                 poro, rcond, hcap, depth, volume, dx = None, dy = None,
                 dz = None, kx=None,ky = None, kz =None,
                 op_num=None, cache: bool= False):

        super().__init__(timer, cache)
        # connectivity
        self.cell_m = np.asarray(cell_m, dtype= int)
        self.cell_p = np.asarray(cell_p, dtype= int)
        self.tran = np.asarray(tran, dtype= float)
        self.tran_thermal = np.asarray(tran_thermal, dtype= float)

        # properties
        self.poro = np.asarray(poro, dtype=float)
        self.rcond = np.asarray(rcond, dtype=float)
        self.hcap = np.asarray(hcap, dtype=float)
        self.depth = np.asarray(depth, dtype=float)
        self.volume = np.asarray(volume, dtype=float)
        if op_num is None:
            self.op_num = np.zeros_like(self.poro, dtype= int)
        else:
            self.op_num = np.asarray(op_num, dtype= int)

        self.n = self.poro.size
        self.ndims = 2 # currently nz =1
        self.actnum = np.ones(self.n, dtype= bool)
        self.global_data = {}

        self.dx = np.asarray(dx, dtype=float) if dx is not None else None
        self.dy = np.asarray(dy, dtype=float) if dy is not None else None
        self.dz = np.asarray(dz, dtype=float) if dz is not None else None
        self.kx = np.asarray(kx, dtype=float) if kx is not None else None
        self.ky = np.asarray(ky, dtype=float) if ky is not None else None
        self.kz = np.asarray(kz, dtype=float) if kz is not None else None


    def discretize(self, cache: bool = False, verbose: bool = False) -> conn_mesh:

        """build conn_mesh from pre-assembled data"""
        cm = np.asarray(self.cell_m, dtype=np.int32)
        cp = np.asarray(self.cell_p, dtype=np.int32)
        mesh = conn_mesh()
        mesh.init(
            index_vector(cm),
            index_vector(cp),
            value_vector(self.tran),
            value_vector(self.tran_thermal))

        # Create numpy arrays wrapped around mesh data
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.depth, copy=False)[:] = self.depth

        np.array(mesh.volume, copy=False)[:] = self.volume
        np.array(mesh.op_num, copy=False)[:] = self.op_num

        self.mesh = mesh

        return mesh

    def cal_well_index_from_cell(self,cell_index: int, well_radius: float = 0.0762,segment_direction='z_axis',skin=0) -> float:
        dx = float(self.dx[cell_index])
        dy = float(self.dy[cell_index])
        dz = float(self.dz[cell_index])
        kx = float(self.kx[cell_index])
        ky = float(self.ky[cell_index])
        kz = float(self.kz[cell_index])
        darcy_constant = 0.0085267146719160104986876640419948
        if segment_direction == 'z_axis':
            if kx * ky != 0:
                peaceman_rad = (
                        0.28
                        * np.sqrt(np.sqrt(ky / kx) * dx**2 + np.sqrt(kx / ky) * dy**2)
                        / ((ky / kx) ** (1 / 4) + (kx / ky) ** (1 / 4))
                    )
                well_index = (
                        2
                        * np.pi
                        * dz
                        * np.sqrt(kx * ky)
                        / (np.log(peaceman_rad / well_radius) + skin)
                    )

                conduction_rad = 0.28 * np.sqrt(dx**2 + dy**2) / 2.0
                well_indexD = (
                    2 * np.pi * dz / (np.log(conduction_rad / well_radius) + skin))
        elif segment_direction == 'x_axis':
            if kz * ky != 0:
                peaceman_rad = (
                        0.28
                        * np.sqrt(np.sqrt(ky / kz) * dz**2 + np.sqrt(kz / ky) * dy**2)
                        / ((ky / kz) ** (1 / 4) + (kz / ky) ** (1 / 4))
                    )
                well_index = (
                        2
                        * np.pi
                        * dx
                        * np.sqrt(kz * ky)
                        / (np.log(peaceman_rad / well_radius) + skin)
                    )

                conduction_rad = 0.28 * np.sqrt(dz**2 + dy**2) / 2.0
                well_indexD = (
                        2 * np.pi * dx / (np.log(conduction_rad / well_radius) + skin)
                    )
        elif segment_direction == 'y_axis':
            if kx * kz != 0:
                peaceman_rad = (
                        0.28
                        * np.sqrt(np.sqrt(kz / kx) * dx**2 + np.sqrt(kx / kz) * dz**2)
                        / ((kz / kx) ** (1 / 4) + (kx / kz) ** (1 / 4))
                    )
                well_index = (
                        2
                        * np.pi
                        * dy
                        * np.sqrt(kx * kz)
                        / (np.log(peaceman_rad / well_radius) + skin)
                    )

                conduction_rad = 0.28 * np.sqrt(dz**2 + dx**2) / 2.0
                well_indexD = (
                        2 * np.pi * dy / (np.log(conduction_rad / well_radius) + skin)
                    )

        well_index = well_index * darcy_constant
        return well_index, well_indexD

    # The issue with this section lies in the fact that once the LGR is located，
    # the global index of the cell containing the well
    # currently requires the user to manually compute both the global index and
    # cell index before passing them to the well block.
    def add_perforation(
        self,
        well_name: str,
        cell_index: int,
        well_radius: float = 0.0762,
        well_index: float = None,
        well_indexD: float = 0.0,
        segment_direction: str = 'z_axis',
        skin: float = 0,
        multi_segment: bool = False,
        verbose: bool = False,
    ):
        well = self.get_well(well_name)
        wi, wid = self.cal_well_index_from_cell(
            cell_index, well_radius, segment_direction, skin
        )

        if well_index is None:
            well_index = wi

        if well_indexD is None:
            well_indexD = wid

        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0

        if len(well.perforations) == 0:
            well.well_head_depth = np.array(self.mesh.depth, copy=False)[cell_index]
            well.well_body_depth = well.well_head_depth
            # I don't consider CPG type
        else:
            well.well_head_depth = min(
                well.well_head_depth,
                np.array(self.mesh.depth, copy=False)[cell_index],
            )
            well.well_body_depth = well.well_body_depth

        for p in well.perforations:
            if p[0] == well_block and p[1] == cell_index:
                raise ValueError(
                    f'Perforation for well {well.name} in block {cell_index} already exists.'
                )
                return

        well.perforations = well.perforations + [
            (well_block, cell_index, well_index, well_indexD)
        ]

        if verbose:
            print(
                f'Added perforation for well {well.name} to block {cell_index} '
                f'with WI={well_index}, WID={well_indexD}'
            )

        assert well_index >= 0
        assert well_indexD >= 0

        return
