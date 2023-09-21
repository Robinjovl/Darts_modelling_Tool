from reservoir import UnstructReservoir, nabla_ref1
import numpy as np
import os

def test_compare_discretizers(mesh='rect'):
    pm_reservoir = UnstructReservoir(discretizer='pm_discretizer', mesh=mesh)
    new_reservoir = UnstructReservoir(discretizer='new_discretizer', mesh=mesh)

    for i in range(pm_reservoir.unstr_discr.mat_cells_tot):
        old_grad = pm_reservoir.get_gradients_pm_discretizer(i)
        new_grad = new_reservoir.get_gradients_new_discretizer(i)
        x = np.append(np.array(new_reservoir.discr_mesh.centroids[i].values, copy=False), 0.0)
        true_grad = nabla_ref1(x)[:,:3].flatten()
        assert((np.fabs(old_grad - true_grad) < 1.e-4 * true_grad).all())
        assert((np.fabs(new_grad - true_grad) < 1.e-4 * true_grad).all())

    print('OK: ' + mesh)





test_compare_discretizers(mesh='rect')
test_compare_discretizers(mesh='tetra')

