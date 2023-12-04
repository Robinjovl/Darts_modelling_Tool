from reservoir import UnstructReservoir
import numpy as np
import os

def test_compare_discretizers(mesh='rect', thermal=False, abs_tol=1e-2, rel_tol=1e-4):
    n_dim = 3
    pm_reservoir = UnstructReservoir(discretizer='pm_discretizer', mesh=mesh)
    new_reservoir = UnstructReservoir(discretizer='new_discretizer', mesh=mesh, thermal=thermal)

    # # check gradients
    for i in range(pm_reservoir.unstr_discr.mat_cells_tot):
        new_grad = new_reservoir.get_gradients_new_discretizer(i)
        x = np.append(np.array(new_reservoir.discr_mesh.centroids[i].values, copy=False), 0.0)
        true_grad = new_reservoir.nabla_ref1(x)[:,:n_dim].flatten()
        assert np.isclose(new_grad, true_grad, rtol=rel_tol, atol=abs_tol).all()
        if not thermal:
            old_grad = pm_reservoir.get_gradients_pm_discretizer(i)
            assert np.isclose(old_grad, true_grad, rtol=rel_tol, atol=abs_tol).all()


    print('OK: gradients, ' + mesh)

    # # check fluxes
    # old approximations
    old_cell_m = np.array(pm_reservoir.pm.cell_m, copy=False)
    old_cell_p = np.array(pm_reservoir.pm.cell_p, copy=False)
    # new approximations
    new_cell_m = np.array(new_reservoir.discr.cell_m, copy=False)
    new_cell_p = np.array(new_reservoir.discr.cell_p, copy=False)
    new_conn_ids = np.array(new_reservoir.discr_mesh.adj_matrix, copy=False)
    new_conns = np.array(new_reservoir.discr_mesh.conns, copy=False)
    # new->old connections mapping
    sum_new = np.array(new_cell_m, dtype=np.float64) + np.array(new_cell_p, dtype=np.float64) / 10 ** 6
    sum_old = np.array(old_cell_m, dtype=np.float64) + np.array(old_cell_p, dtype=np.float64) / 10 ** 6
    inds = np.nonzero(sum_new[:,None] == sum_old)[1]
    for id_new, id_old in enumerate(inds):
        assert(old_cell_m[id_old] == new_cell_m[id_new])
        assert(old_cell_p[id_old] == new_cell_p[id_new])
        conn = new_conns[new_conn_ids[id_new]]
        n = np.array(conn.n.values)
        dx = np.array(conn.c.values) - np.array(new_reservoir.discr_mesh.centroids[new_cell_m[id_new]].values)
        sign = 1.0 if dx.dot(n) > 0 else -1.0
        x = np.append(np.array(conn.c.values), 0.0) # coordinates + time
        x_cell1 = np.append(np.array(new_reservoir.discr_mesh.centroids[new_cell_m[id_new]].values), 0.0) # coordinates + time
        # analytical fluxes
        hooke_an, biot_an, darcy_an, vol_strain_an = new_reservoir.get_analytical_fluxes(x, sign * n, x_cell1)
        # old fluxes
        hooke_old, biot_old, darcy_old, vol_strain_old = pm_reservoir.get_fluxes_pm_discretizer(id_old)
        hooke_old, biot_old, darcy_old, vol_strain_old = hooke_old / conn.area, biot_old / conn.area, \
                                                         darcy_old / conn.area, vol_strain_old / conn.area
        # new fluxes
        hooke_new, biot_new, darcy_new, vol_strain_new = new_reservoir.get_fluxes_new_discretizer(id_new)
        hooke_new, biot_new, darcy_new, vol_strain_new = hooke_new / conn.area, biot_new / conn.area, \
                                                         darcy_new / conn.area, vol_strain_new / conn.area
        # check Hooke's (effective) traction
        assert np.isclose(hooke_old, hooke_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(hooke_new, hooke_an, rtol=rel_tol, atol=abs_tol).all()
        # check Biot's term in traction
        assert np.isclose(biot_old, biot_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(biot_new, biot_an, rtol=rel_tol, atol=abs_tol).all()
        # check Darcy fluxes
        assert np.isclose(darcy_old, darcy_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(darcy_new, darcy_an, rtol=rel_tol, atol=abs_tol).all()
        # check Biot's term (~ volumentric strains) in fluid fluxes
        assert np.isclose(vol_strain_old, vol_strain_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(vol_strain_new, vol_strain_an, rtol=rel_tol, atol=abs_tol).all()
        # check Fick's term
        # check Fourier's term

    print('OK: fluxes, ' + mesh)


#test_compare_discretizers(mesh='rect', abs_tol=1e-8, rel_tol=1e-8)
#test_compare_discretizers(mesh='tetra', abs_tol=1e-8, rel_tol=1e-8)

test_compare_discretizers(mesh='rect',  thermal=True, abs_tol=1e-8, rel_tol=1e-8)
test_compare_discretizers(mesh='tetra', thermal=True, abs_tol=1e-8, rel_tol=1e-8)