"""Elastic column reservoir for the 1D wave-propagation validation (thesis Sec. 6.4.1).

A box [0, a] x [0, a] x [0, H] discretized with a single column of hexahedral
cells along z (pm_discretizer). All boundaries are rollers except the top one
(BND_Z+), where a time-dependent normal displacement u_z(t) is prescribed
through a STUCK_ROLLER condition whose value is updated every time step.
"""

import numpy as np

from darts.discretizer import elem_loc
from darts.engines import pm_discretizer
from darts.input.input_data import InputData
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.unstruct_reservoir_mech import (
    UnstructReservoirMech,
    get_lambda_mu,
    set_domain_tags,
)


class ColumnReservoir(UnstructReservoirMech):
    def __init__(self, timer, idata: InputData, fluid_vars):
        super().__init__(timer, discretizer='pm_discretizer', fluid_vars=fluid_vars, thermoporoelasticity=False)
        self.bnd_tags = idata.mesh.bnd_tags
        self.domain_tags = set_domain_tags(matrix_tags=idata.mesh.matrix_tags, bnd_tags=list(self.bnd_tags.values()))
        self.mesh_filename = idata.mesh.mesh_filename
        self.top_tag = self.bnd_tags['BND_Z+']

        self.set_uniform_initial_conditions(idata=idata)
        physical_tags = {'matrix': list(self.domain_tags[elem_loc.MATRIX])}
        physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_filename, physical_tags=physical_tags)
        self.unstr_discr.eps_t = 1.E+0
        self.unstr_discr.eps_n = 1.E+0
        self.unstr_discr.mu = 3.2
        self.unstr_discr.P12 = 0
        self.unstr_discr.Prol = 1
        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3

        self.lam, self.mu = get_lambda_mu(idata.rock.E, idata.rock.nu)
        self.init_matrix_stiffness({self.unstr_discr.physical_tags['matrix'][0]:
                                    {'E': idata.rock.E, 'nu': idata.rock.nu, 'stiffness': idata.rock.stiffness}})
        self.set_boundary_conditions(idata)
        self.unstr_discr.load_mesh(permx=1, permy=1, permz=1, frac_aper=0)
        self.unstr_discr.calc_cell_neighbours()

        self.pm = pm_discretizer()
        self.set_scheme_pm_discretizer()
        self.pm.neumann_boundaries_grad_reconstruction = True
        self.init_gravity(gravity_on=False)

        self.init_faces_centers_pm_discretizer()
        self.init_uniform_properties(idata=idata)
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()

        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_faces_tot

        self.init_reservoir_main(idata=idata)
        self.set_pzt_bounds(p=self.p_init, z=None, t=self.t_init)

        # cell centroids along the column axis
        self.z_centers = np.array([self.unstr_discr.mat_cell_info_dict[i].centroid[2] for i in range(self.n_matrix)])
        self.H = np.max(self.unstr_discr.mesh_data.points[:, 2])

    def set_boundary_conditions(self, idata):
        self.boundary_conditions = idata.boundary
        self.bnd_tags = idata.mesh.bnd_tags
        self.set_boundary_conditions_pm_discretizer()

    def set_top_displacement(self, uz: float):
        """Prescribe the normal displacement at the top boundary (z = H).

        The outward normal of the top face is +z, so 'rn' = uz directly.
        Must be followed by update_trans() so that mesh.bc picks the new value.
        """
        self.boundary_conditions[self.top_tag]['mech'] = self.bc_type.STUCK_ROLLER(uz)
        self.set_boundary_conditions_pm_discretizer()
        self.init_bc_rhs()
