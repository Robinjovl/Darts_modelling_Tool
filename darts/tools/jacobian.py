import pickle

import numpy as np
from scipy.sparse import bsr_matrix

from darts.models.darts_model import DartsModel


def write_jacobian_to_pkl(m: DartsModel, filename: str):
    # get current jacobian and rhs from the engine
    jac_rows = np.asarray(m.physics.engine.jac_rows)
    jac_cols = np.asarray(m.physics.engine.jac_cols)
    jac_diag = np.asarray(m.physics.engine.jac_diags)
    jac_vals = np.asarray(m.physics.engine.jac_vals)

    n_res = m.reservoir.mesh.n_res_blocks * m.physics.n_vars
    rhs = np.array(m.physics.engine.RHS, copy=False)[:n_res]

    # make a dictionary
    jac = {
        'rows': jac_rows,
        'cols': jac_cols,
        'diag': jac_diag,
        'vals': jac_vals,
        'rhs': rhs,
    }

    # save to PKL file
    with open(filename, 'wb') as f:
        pickle.dump(jac, f)


def read_jacobian_from_pkl(m, filename):
    # load pkl to dict
    with open(filename, 'rb') as f:
        j = pickle.load(f)

    # extract arrays from dict
    jac_rows = j['rows']
    jac_cols = j['cols']
    jac_diag = j['diag']
    jac_vals = j['vals']
    jac_rhs = j['rhs']
    n = jac_diag.size  # n rows
    nonzeros = jac_cols.size
    b = int(np.sqrt(jac_vals.size / nonzeros))
    jac_vals = jac_vals.reshape(nonzeros, b, b)

    # create scipy matrix from arrays
    mat = bsr_matrix((jac_vals, jac_cols, jac_rows))
    return mat


def plot_bcsr_matrix(mat, filename='mat.png'):
    import matplotlib.pyplot as plt

    plt.spy(mat)
    plt.savefig(filename)
    plt.close()
