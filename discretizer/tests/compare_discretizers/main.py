from reservoir import UnstructReservoir, nabla_ref1
import numpy as np
import os

def test_compare_discretizers(mesh='rect'):
    pm_reservoir = UnstructReservoir(discretizer='pm_discretizer', mesh=mesh)
    new_reservoir = UnstructReservoir(discretizer='new_discretizer', mesh=mesh)
    n_dim = 3

    # # check gradients
    # for i in range(pm_reservoir.unstr_discr.mat_cells_tot):
    #
    #     old_grad = pm_reservoir.get_gradients_pm_discretizer(i)
    #     new_grad = new_reservoir.get_gradients_new_discretizer(i)
    #     x = np.append(np.array(new_reservoir.discr_mesh.centroids[i].values, copy=False), 0.0)
    #     true_grad = nabla_ref1(x)[:,:n_dim].flatten()
    #     assert((np.fabs(old_grad - true_grad) < 1.e-4 * true_grad).all())
    #     assert((np.fabs(new_grad - true_grad) < 1.e-4 * true_grad).all())
    #
    # print('OK: gradients, ' + mesh)

    # # check fluxes
    # old approximations
    old_cell_m = np.array(pm_reservoir.pm.cell_m, copy=False)
    old_cell_p = np.array(pm_reservoir.pm.cell_p, copy=False)
    # new approximations
    new_cell_m = np.array(new_reservoir.discr.cell_m, copy=False)
    new_cell_p = np.array(new_reservoir.discr.cell_p, copy=False)
    new_conn_ids = np.array(new_reservoir.discr_mesh.adj_matrix, copy=False)
    new_conns = np.array(new_reservoir.discr_mesh.conns, copy=False)
    # connection correspondence
    sum_new = np.array(new_cell_m, dtype=np.float64) + np.array(new_cell_p, dtype=np.float64) / 10 ** 6
    sum_old = np.array(old_cell_m, dtype=np.float64) + np.array(old_cell_p, dtype=np.float64) / 10 ** 6
    inds = np.nonzero(sum_new[:,None] == sum_old)[1]
    for id_new, id_old in enumerate(inds):
        assert(old_cell_m[id_old] == new_cell_m[id_new])
        assert(old_cell_p[id_old] == new_cell_p[id_new])
        conn = new_conns[new_conn_ids[id_new]]
        sign = 1.0 if new_cell_m[id_old] == conn.elem_id1 else -1.0
        x = np.append(np.array(conn.c.values), 0.0) # coordinates + time
        # fluxes
        hooke_an = new_reservoir.get_analytical_fluxes(x, conn.n.values)
        hooke_new = sign * new_reservoir.get_fluxes_new_discretizer(id_new) / conn.area
        hooke_old = pm_reservoir.get_fluxes_pm_discretizer(id_old) / conn.area
        aa = 55
    print('OK: fluxes, ' + mesh)





test_compare_discretizers(mesh='rect')
test_compare_discretizers(mesh='tetra')

