import numpy as np
import pickle

#orig_cache_filename = 'pm.pkl'
#new_cache_filename = 'mech_discr.pkl'

def compare_gradients(orig_cache_filename=None, new_cache_filename=None, orig_pm_arg=None, new_pm_arg=None):
    if orig_cache_filename is not None:
        with open(orig_cache_filename, "rb") as fp:
            orig_pm = pickle.load(fp)
    else:
        orig_pm = orig_pm_arg

    if new_cache_filename is not None:
        with open(new_cache_filename, "rb") as fp:
            new_pm = pickle.load(fp)
    else:
        new_pm = new_pm_arg

    # n cells
    n = np.array(orig_pm.cell_centers,copy=False).shape[0]

    # compute max difference by cells
    diff_index_max = 0
    diff_values_max = 0
    diff_values_rel_max = 0
    vals_max = vals2_max = 0
    eps = 1e-12
    for i in range(n):

        vals2 = np.array(new_pm.p_grads[i].a.values, copy=False)
        ind2 = np.array(new_pm.p_grads[i].stencil, copy=False)
        #self.discr.u_grads[0]
        ind2 = np.array(ind2)
        vals2 = np.array(vals2)

        ind, vals = orig_pm.get_gradient(i)
        ind = np.array(ind)
        vals = np.array(vals)

        diff_index_max  = max(diff_index_max, (ind - ind2).max())
        diff_values_max = max(diff_values_max,(vals - vals2).max())
        diff_values_rel_max = max(diff_values_rel_max, diff_values_max / (vals.max() + vals2.max() + eps))
        vals_max  = max(vals_max, vals.max())
        vals2_max = max(vals2_max, vals2.max())

    print('diff_index_max =', diff_index_max)
    print('diff_values_max =', diff_values_max)
    print('diff_values_rel_max =', diff_values_rel_max)
    print('vals_max =', vals_max, 'vals2_max =', vals2_max)