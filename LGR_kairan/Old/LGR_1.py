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

    def set_reservoir(self):
        nx = 5
        ny = 3
        self.refined_cells_ij = [(2,2),(4,2)]  # (i,j) of coarse grid to be refined
        actnum0 = np.ones(nx * ny, dtype=np.int32)
        for i_c,j_c in self.refined_cells_ij:
            g = (j_c-1)*nx + (i_c-1)
            actnum0[g] = 0  # deactivate the coarse cell that will be refined

        self.level0 = StructReservoir(self.timer, nx=nx, ny=ny, nz=1, dx=100, dy=100, dz=5,
                                         permx=50, permy=50, permz=50, poro=0.1, depth=2000,actnum=actnum0)
        self.level1_i = StructReservoir(self.timer, nx=3, ny=3, nz=1, dx=100/3, dy=100/3, dz=5,
                                        permx=50, permy=50, permz=50, poro=0.1, depth=2000)
        self.level1_p = StructReservoir(self.timer, nx=3, ny=3, nz=1, dx=100/3, dy=100/3, dz=5,
                                        permx=50, permy=50, permz=50, poro=0.1, depth=2000)

        dx_imag = np.array([100,100/3,100/3,100/3,100],dtype=float)
        dy_imag = np.array([100,100/3,100/3,100/3,100],dtype=float)
        self.level1_imag_inj = StructReservoir(self.timer, nx=5, ny=5, nz=1, dx=dx_imag, dy=dy_imag, dz=5,
                                        permx=50, permy=50, permz=50, poro=0.1, depth=2000)
        self.level1_imag_prd = StructReservoir(self.timer, nx=5, ny=5, nz=1, dx=dx_imag, dy=dy_imag, dz=5,
                                        permx=50, permy=50, permz=50, poro=0.1, depth=2000)


        cm_all, cp_all, T_all, T_all_therm = self.assemble_lgr_connections()

       # assemble properties
        disc0 = self.level0.discretizer
        l2g0 = disc0.local_to_global          # l2g0 has already eliminated inactive cells

        dx0 = disc0.convert_to_flat_array(self.level0.global_data['dx'], 'dx')[l2g0]
        dy0 = disc0.convert_to_flat_array(self.level0.global_data['dy'], 'dy')[l2g0]
        dz0 = disc0.convert_to_flat_array(self.level0.global_data['dz'], 'dz')[l2g0]
        kx0 = disc0.convert_to_flat_array(self.level0.global_data['permx'], 'permx')[l2g0]
        ky0 = disc0.convert_to_flat_array(self.level0.global_data['permy'], 'permy')[l2g0]
        kz0 = disc0.convert_to_flat_array(self.level0.global_data['permz'], 'permz')[l2g0]

        disc1_i = self.level1_i.discretizer
        disc1_p = self.level1_p.discretizer
        dx = np.concatenate([dx0,
                             disc1_i.convert_to_flat_array(self.level1_i.global_data['dx'], 'dx'),
                             disc1_p.convert_to_flat_array(self.level1_p.global_data['dx'], 'dx')])
        dy = np.concatenate([dy0,
                             disc1_i.convert_to_flat_array(self.level1_i.global_data['dy'], 'dy'),
                             disc1_p.convert_to_flat_array(self.level1_p.global_data['dy'], 'dy')])
        dz = np.concatenate([dz0,
                             disc1_i.convert_to_flat_array(self.level1_i.global_data['dz'], 'dz'),
                             disc1_p.convert_to_flat_array(self.level1_p.global_data['dz'], 'dz')])
        kx = np.concatenate([kx0,
                             disc1_i.convert_to_flat_array(self.level1_i.global_data['permx'], 'permx'),
                             disc1_p.convert_to_flat_array(self.level1_p.global_data['permx'], 'permx')])
        ky = np.concatenate([ky0,
                             disc1_i.convert_to_flat_array(self.level1_i.global_data['permy'], 'permy'),
                             disc1_p.convert_to_flat_array(self.level1_p.global_data['permy'], 'permy')])
        kz = np.concatenate([kz0,
                             disc1_i.convert_to_flat_array(self.level1_i.global_data['permz'], 'permz'),
                             disc1_p.convert_to_flat_array(self.level1_p.global_data['permz'], 'permz')])

        # coarse properties in L0 order
        vol_g0 = np.ones(self.level0.nx*self.level0.ny*self.level0.nz, dtype=float) * (100*100*5)
        volume0 = vol_g0[l2g0]

        poro0  = disc0.convert_to_flat_array(self.level0.global_data['poro'],  'poro')[l2g0]
        depth0 = disc0.convert_to_flat_array(self.level0.global_data['depth'], 'depth')[l2g0]
        # poro_i = 0.1 * np.ones(self.level1_i.n)
        poro_i = disc1_i.convert_to_flat_array(self.level1_i.global_data['poro'], 'poro')
        poro_p = disc1_p.convert_to_flat_array(self.level1_p.global_data['poro'], 'poro')
        #poro_p = 0.1 * np.ones(self.level1_p.n)
        depth_i = disc1_i.convert_to_flat_array(self.level1_i.global_data['depth'], 'depth')
        depth_p = disc1_p.convert_to_flat_array(self.level1_p.global_data['depth'], 'depth')
        # depth_i = 2000.0 * np.ones(self.level1_i.n)
        # depth_p = 2000.0 * np.ones(self.level1_p.n)
        volume_i = (100/3)*(100/3)*5 * np.ones(self.level1_i.n)
        volume_p = (100/3)*(100/3)*5 * np.ones(self.level1_p.n)

        poro = np.concatenate([poro0, poro_i, poro_p])
        rcon = np.zeros_like(poro)
        hcap = np.zeros_like(poro)
        depth = np.concatenate([depth0, depth_i, depth_p])
        volume = np.concatenate([volume0, volume_i, volume_p])


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
        return

    def assemble_lgr_connections(self):
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
        # # To find the index of cell that is refined
        # nx0 = self.level0.nx
        # ny0 = self.level0.ny
        # refined_cells_ij = [(2,2),(4,2)]
        # refined_idx_list = []
        # for i_c,j_c in refined_cells_ij:
        #     idx = (j_c-1)*nx0 + (i_c-1)
        #     refined_idx_list.append(idx)

        # refined_idx = np.array(refined_idx_list, dtype=int)

        # # Eliminate all connection related to refined cell
        # mask_keep = (~np.isin(cm0,refined_idx)) & (~np.isin(cp0,refined_idx))
        # cm0 = cm0[mask_keep]
        # cp0 = cp0[mask_keep]
        # T0 = T0[mask_keep]
        # T0_therm = T0_therm[mask_keep]

        # step 2 discretize fine cells of level 1
        self.level1_i.discretize()
        disc1_i = self.level1_i.discretizer
        cm1_i, cp1_i, T1_i, T1_therm_i = disc1_i.calc_structured_discr()

        self.level1_p.discretize()
        disc1_p = self.level1_p.discretizer
        cm1_p, cp1_p, T1_p, T1_therm_p = disc1_p.calc_structured_discr()

        # it needs a mapping to transfer local index to global index
        # here it numbers index from last index of coarse grid
        fine_global_offset_inj = n0_act
        cm1_i_global = cm1_i + fine_global_offset_inj
        cp1_i_global = cp1_i + fine_global_offset_inj

        fine_global_offset_prd = n0_act+ self.level1_i.n
        cm1_p_global = cm1_p + fine_global_offset_prd
        cp1_p_global = cp1_p + fine_global_offset_prd

        # STEP 3: discretize imaginary grid,
        #in this case, refinements around injection well and production well are same
        self.level1_imag_inj.discretize()
        disc_im = self.level1_imag_inj.discretizer
        cmi, cpi, Ti, Ti_therm = disc_im.calc_structured_discr()
        nxim, nyim = self.level1_imag_inj.nx, self.level1_imag_inj.ny

        # indices of boundary cells of area that we refined
        fine_im_indices_list = []
        for j in range(1,4):
            for i in range(1,4):
                fine_im_indices_list.append(j*nxim+i)
        fine_im_indices = np.array(fine_im_indices_list,dtype=int)
        fine_im_set = set(fine_im_indices.tolist())

        coarse_im_indices = []
        for i in range(1,4):
            coarse_im_indices.append(i)
        for i in range(1,4):
            coarse_im_indices.append(4*nxim + i)

        for j in range(1,4):
            coarse_im_indices.append(j*nxim)
        for j in range(1,4):
            coarse_im_indices.append(j*nxim + 4)
        coarse_im_indices = np.array(coarse_im_indices,dtype=int)
        coarse_im_set = set(coarse_im_indices.tolist())
        # sample the connection we want
        is_fine_coarse = np.array([(cm in fine_im_set and cp in coarse_im_set)
                                   or (cp in fine_im_set and cm in coarse_im_set)
                                   for cm, cp in zip(cmi, cpi)], dtype= bool,)
        # boundary connection (fine-coarse)
        cmi_fc = cmi[is_fine_coarse]
        cpi_fc = cpi[is_fine_coarse]
        Ti_fc = Ti[is_fine_coarse]
        Ti_therm_fc = Ti_therm[is_fine_coarse]

        # Map the fine index of imaginary to the global index of the actual fine block
        # map the coarse index to the real index of level 0
        imag_fine_to_local = {}
        k = 0
        for j in range(1,4):
            for i in range(1,4):
                imag_idx = j*nxim + i # index of 5*5 grid
                imag_fine_to_local[imag_idx] = k # index of 3*3 grid
                k += 1
        # coarse neighbor global index before eliminate inactive cell
        nbr_inj_g = {'left': (self.refined_cells_ij[0][0]-1)-1 + (self.refined_cells_ij[0][1]-1)*self.level0.nx,
                      'right': (self.refined_cells_ij[0][0]+1)-1 + (self.refined_cells_ij[0][1]-1)*self.level0.nx,
                      'up': (self.refined_cells_ij[0][0])-1 + (self.refined_cells_ij[0][1]-1-1)*self.level0.nx,
                      'down': (self.refined_cells_ij[0][0])-1 + (self.refined_cells_ij[0][1]+1-1)*self.level0.nx}

        nbr_prd_g = {'left': (self.refined_cells_ij[1][0]-1)-1 + (self.refined_cells_ij[1][1]-1)*self.level0.nx,
                        'right': (self.refined_cells_ij[1][0]+1)-1 + (self.refined_cells_ij[1][1]-1)*self.level0.nx,
                        'up': (self.refined_cells_ij[1][0])-1 + (self.refined_cells_ij[1][1]-1-1)*self.level0.nx,
                        'down': (self.refined_cells_ij[1][0])-1 + (self.refined_cells_ij[1][1]+1-1)*self.level0.nx}
        # map to new coarse local indices (after actnum squeeze)
        nbr_inj_l = {k: int(g2l0[v]) for k, v in nbr_inj_g.items()}
        nbr_prd_l = {k: int(g2l0[v]) for k, v in nbr_prd_g.items()}

        left_im = [j * nxim for j in range(1,4)]
        right_im = [4 + j*nxim for j in range(1,4)]
        up_im = [i for i in range(1,4)]
        down_im = [4*nxim + i for i in range(1,4)]
        image_coarse_to_level0_inj = {}
        # the index change around injection well
        for idx in left_im:
            image_coarse_to_level0_inj[idx] = nbr_inj_l['left']
        for idx in right_im:
            image_coarse_to_level0_inj[idx] = nbr_inj_l['right']
        for idx in up_im:
            image_coarse_to_level0_inj[idx] = nbr_inj_l['up']
        for idx in down_im:
            image_coarse_to_level0_inj[idx] = nbr_inj_l['down']

        # index change around production well
        image_coarse_to_level0_prd = {}
        for idx in left_im:
            image_coarse_to_level0_prd[idx] = nbr_prd_l['left']
        for idx in right_im:
            image_coarse_to_level0_prd[idx] = nbr_prd_l['right']
        for idx in up_im:
            image_coarse_to_level0_prd[idx] = nbr_prd_l['up']
        for idx in down_im:
            image_coarse_to_level0_prd[idx] = nbr_prd_l['down']

        cm_fc_global = []
        cp_fc_global = []
        T_fc_global = []
        T_therm_fc_global = []
        for cm, cp, t, tt in zip(cmi_fc,cpi_fc, Ti_fc, Ti_therm_fc):
            if cm in imag_fine_to_local and cp in image_coarse_to_level0_inj:
                fine_local = imag_fine_to_local[cm]
                #Mapped the surrounding grid indices of imag to the indices within level0
                coarse_level0_l_id = image_coarse_to_level0_inj[cp]
            elif cp in imag_fine_to_local and cm in image_coarse_to_level0_inj:
                 #Convert the index of the 3×3 region in imag to the index in step 2.
                fine_local = imag_fine_to_local[cp]
                coarse_level0_l_id = image_coarse_to_level0_inj[cm]
            else:
                continue
            fine_global = fine_local + fine_global_offset_inj
            cm_fc_global.append(coarse_level0_l_id)
            cp_fc_global.append(fine_global)
            T_fc_global.append(t)
            T_therm_fc_global.append(tt)

        #repeat preceding process for production well
        for cm,cp,t,tt in zip(cmi_fc,cpi_fc, Ti_fc, Ti_therm_fc):
            if cm in imag_fine_to_local and cp in image_coarse_to_level0_prd:
                fine_local = imag_fine_to_local[cm]
                coarse_level0_l_id = image_coarse_to_level0_prd[cp]
            elif cp in imag_fine_to_local and cm in image_coarse_to_level0_prd:
                fine_local = imag_fine_to_local[cp]
                coarse_level0_l_id = image_coarse_to_level0_prd[cm]
            else:
                continue
            fine_global = fine_local + fine_global_offset_prd
            cm_fc_global.append(coarse_level0_l_id)
            cp_fc_global.append(fine_global)
            T_fc_global.append(t)
            T_therm_fc_global.append(tt)

        cm_fc_global = np.array(cm_fc_global,dtype=int)
        cp_fc_global = np.array(cp_fc_global, dtype=int)

        T_fc_global = np.array(T_fc_global,dtype=float)
        T_therm_fc_global = np.array(T_therm_fc_global,dtype=float)

        cm_all = np.concatenate([cm0, cm1_i_global, cm1_p_global, cm_fc_global])
        cp_all = np.concatenate([cp0, cp1_i_global, cp1_p_global, cp_fc_global])
        T_all = np.concatenate([T0,T1_i,T1_p, T_fc_global])
        T_all_therm = np.concatenate([T0_therm,T1_therm_i,T1_therm_p, T_therm_fc_global])
        print(f'cm is {cm_all}')
        print(f'cp is {cp_all}')

        return cm_all, cp_all, T_all, T_all_therm


    def set_wells(self):
        self.reservoir.add_well("I1")
        center_local = 4
        n0_act = len(self.level0.discretizer.local_to_global)
        n_inj = self.level1_i.n

        inj_global = n0_act + center_local
        prd_global = n0_act + n_inj + center_local
        self.reservoir.add_perforation("I1", cell_index=inj_global)

        self.reservoir.add_well("P1")
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
