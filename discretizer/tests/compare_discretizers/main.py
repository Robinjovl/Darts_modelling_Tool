from reservoir import UnstructReservoir
import numpy as np
import os, sys, time

original_stdout = os.dup(1)

# add path to import
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)  # 1 level up
parent_dir2 = os.path.dirname(parent_dir)  # 2 levels up
parent_dir3 = os.path.dirname(parent_dir2)  # 3 levels up
models_dir = os.path.join(parent_dir3, 'models')
sys.path.insert(0, models_dir)

from for_each_model import abort_redirection, redirect_all_output

def compare(x_1, x_2, name_1, name_2, rel_tol=1e-8, abs_tol=1e-8):
    eps = 1e-12
    close = np.isclose(x_1, x_2, rtol=rel_tol, atol=abs_tol).all()
    if not close:
        print('Arrays ', name_1, ' and ', name_2, ' differ! ')
        print('   ', name_1, ': ', x_1)
        print('   ', name_2, ': ', x_2)
        print('    abs.diff:', np.fabs(x_1 - x_2))
        print('    rel.diff:', np.fabs((x_1 - x_2) / (np.maximum(x_1, x_2) + eps)))
        return 1
    return 0

def test_compare_discretizers(mesh='rect', thermal=False, abs_tol=1e-8, rel_tol=1e-8):
    n_dim = 3
    pm_reservoir = UnstructReservoir(discretizer='pm_discretizer', mesh=mesh)
    new_reservoir = UnstructReservoir(discretizer='new_discretizer', mesh=mesh, thermal=thermal)

    # check gradients
    diff_grad_flag = 0
    for i in range(pm_reservoir.unstr_discr.mat_cells_tot):
        new_grad = new_reservoir.get_gradients_new_discretizer(i)
        x = np.append(np.array(new_reservoir.discr_mesh.centroids[i].values), 0.0)
        true_grad = new_reservoir.nabla_ref1(x)[:,:n_dim].flatten()
        diff_grad_flag += compare(new_grad, true_grad, 'new_grad', 'true_grad')
        if not thermal:
            old_grad = pm_reservoir.get_gradients_pm_discretizer(i)
            diff_grad_flag += compare(old_grad, true_grad, 'old_grad', 'true_grad')
        if diff_grad_flag:
            return 1

    if not diff_grad_flag:
        print('OK: gradients, ' + mesh)
    else:
        print('ERR: gradients, ' + mesh)

    # check fluxes
    diff_fluxes_flag = 0
    # old approximations
    old_cell_m = np.array(pm_reservoir.pm.cell_m, copy=False)
    old_cell_p = np.array(pm_reservoir.pm.cell_p, copy=False)
    # new approximations
    new_cell_m = np.array(new_reservoir.discr.cell_m, copy=False)
    new_cell_p = np.array(new_reservoir.discr.cell_p, copy=False)
    new_conn_ids = np.array(new_reservoir.discr_mesh.adj_matrix, copy=False)
    new_conns = np.array(new_reservoir.discr_mesh.conns)
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
        diff_fluxes_flag += compare(hooke_old, hooke_an, 'hooke_old', 'hooke_an')
        diff_fluxes_flag += compare(hooke_new, hooke_an, 'hooke_new', 'hooke_an')
        # check Biot's term in traction
        diff_fluxes_flag += compare(biot_old, biot_an, 'biot_old', 'biot_an')
        diff_fluxes_flag += compare(biot_new, biot_an, 'biot_new', 'biot_an')
        # check Darcy fluxes
        diff_fluxes_flag += compare(darcy_old, darcy_an, 'darcy_old', 'darcy_an')
        diff_fluxes_flag += compare(darcy_new, darcy_an, 'darcy_new', 'darcy_an')
        # check Biot's term (~ volumetric strains) in fluid fluxes
        diff_fluxes_flag += compare(vol_strain_old, vol_strain_an, 'vol_strain_old', 'vol_strain_an')
        diff_fluxes_flag += compare(vol_strain_new, vol_strain_an, 'vol_strain_new', 'vol_strain_an')
        #TODO check Fick's term
        if thermal:
            # check Fourier's term (only with analytic)
            diff_fluxes_flag += compare(fourier_new, fourier_an, 'fourier_new', 'fourier_an')
            # check Thermal term (only with analytic)
            diff_fluxes_flag += compare(thermal_new, thermal_an, 'thermal_new', 'thermal_an')
        if diff_fluxes_flag:
            return 1

    if not diff_fluxes_flag:
        print('OK: fluxes, ' + mesh)
    else:
        print('ERR: fluxes, ' + mesh)

    if diff_grad_flag or diff_fluxes_flag:
        return 1
    return 0


# erase previous log file if exists
log_file = os.path.join(os.path.abspath(os.pardir), 'compare_discretizers.log')
f = open(log_file, "w")
f.close()

r1 = r2 = 0
print('Poroelasticity tests...')
starting_time = time.time()
log_stream = redirect_all_output(log_file)
r1 += test_compare_discretizers(mesh='rect',  thermal=False, abs_tol=1e-8, rel_tol=1e-8)
r1 += test_compare_discretizers(mesh='tetra', thermal=False, abs_tol=1e-8, rel_tol=1e-8)
ending_time = time.time()
abort_redirection(log_stream)
str_status = 'OK' if not r1 else 'FAIL'
print('Discretizer Poroelasticity tests: ' + str_status + ', \t%.2f s' % (ending_time - starting_time))

print('Thermoporoelasticity tests...')
log_stream = redirect_all_output(log_file)
starting_time = time.time()
r2 += test_compare_discretizers(mesh='rect',  thermal=True, abs_tol=1e-8, rel_tol=1e-8)
r2 += test_compare_discretizers(mesh='tetra', thermal=True, abs_tol=1e-8, rel_tol=1e-8)
ending_time = time.time()
abort_redirection(log_stream)
str_status = 'OK' if not r1 else 'FAIL'
print('Discretizer Thermoporoelasticity tests: ' + str_status + ', \t%.2f s' % (ending_time - starting_time))

exit(r1 + r2)