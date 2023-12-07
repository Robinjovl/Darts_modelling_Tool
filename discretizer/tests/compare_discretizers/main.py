from reservoir import UnstructReservoir
import numpy as np
import os

def compare(x, y, name, rel_tol=1e-8, abs_tol=1e-8):
    close = np.isclose(x, y, rtol=rel_tol, atol=abs_tol).all()
    if not close:
        print('Arrays ', name, ' differs! ')
        print('    1:', x[:5])
        print('    2:', y[:5])
        return 1
    return 0

def test_compare_discretizers(mesh='rect', thermal=False, abs_tol=1e-8, rel_tol=1e-8):
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
        x_cell1 = np.append(np.array(new_reservoir.discr_mesh.centroids[new_cell_m[id_new]].values), 0.0)  # coordinates + time
        # analytical fluxes
        analytical_fluxes = new_reservoir.get_analytical_fluxes(x, sign * n, x_cell1)
        if new_reservoir.thermal:
            hooke_an, biot_an, darcy_an, vol_strain_an, thermal_an, fourier_an = analytical_fluxes
        else:
            hooke_an, biot_an, darcy_an, vol_strain_an = analytical_fluxes
        # old fluxes
        old_fluxes_1 = pm_reservoir.get_fluxes_pm_discretizer(id_old)
        old_fluxes = list(map(lambda xi: xi / conn.area, old_fluxes_1))
        hooke_old, biot_old, darcy_old, vol_strain_old = old_fluxes
        # new fluxes
        new_fluxes_1 = new_reservoir.get_fluxes_new_discretizer(id_new)
        new_fluxes = list(map(lambda xi: xi / conn.area, new_fluxes_1))
        if new_reservoir.thermal:
            hooke_new, biot_new, darcy_new, vol_strain_new, thermal_new, fourier_new = new_fluxes
        else:
            hooke_new, biot_new, darcy_new, vol_strain_new = new_fluxes
        # check Hooke's (effective) traction
        assert np.isclose(hooke_old, hooke_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(hooke_new, hooke_an, rtol=rel_tol, atol=abs_tol).all()
        # check Biot's term in traction
        assert np.isclose(biot_old, biot_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(biot_new, biot_an, rtol=rel_tol, atol=abs_tol).all()
        # check Darcy fluxes
        assert np.isclose(darcy_old, darcy_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(darcy_new, darcy_an, rtol=rel_tol, atol=abs_tol).all()
        # check Biot's term (~ volumetric strains) in fluid fluxes
        assert np.isclose(vol_strain_old, vol_strain_an, rtol=rel_tol, atol=abs_tol).all()
        assert np.isclose(vol_strain_new, vol_strain_an, rtol=rel_tol, atol=abs_tol).all()
        #TODO check Fick's term

        # check Fourier's term (only with analytic)
        assert np.isclose(fourier_new, fourier_an, rtol=rel_tol, atol=abs_tol).all()
        # check Thermal term (only with analytic)
        assert np.isclose(thermal_new, thermal_an, rtol=rel_tol, atol=abs_tol).all()
    
    print('OK: fluxes, ' + mesh)


#test_compare_discretizers(mesh='rect', abs_tol=1e-8, rel_tol=1e-8)
#test_compare_discretizers(mesh='tetra', abs_tol=1e-8, rel_tol=1e-8)

test_compare_discretizers(mesh='rect',  thermal=True, abs_tol=1e-8, rel_tol=1e-8)
test_compare_discretizers(mesh='tetra', thermal=True, abs_tol=1e-8, rel_tol=1e-8)