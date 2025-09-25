import pickle

import numpy as np
from scipy.sparse import bsr_matrix

from darts.models.darts_model import DartsModel


def check_jacobian(m: DartsModel):
    '''
    Check the current jacobian and rhs from the engine for NaN values.
    :param m: model instance
    :return:
    '''
    # get current jacobian and rhs from the engine
    np.asarray(m.physics.engine.jac_rows)
    np.asarray(m.physics.engine.jac_cols)
    np.asarray(m.physics.engine.jac_diags)
    jac_vals = np.asarray(m.physics.engine.jac_vals)

    m.reservoir.mesh.n_res_blocks * m.physics.n_vars
    rhs = np.array(m.physics.engine.RHS, copy=False)

    has_nan = np.isnan(jac_vals).any()
    assert not has_nan, 'jac_vals has nan'

    has_nan = np.isnan(rhs).any()
    if has_nan:
        nan_indices = np.where(np.isnan(rhs))
        print("rhs indices with nan values:", nan_indices)
    assert not has_nan, 'rhs has nan'


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
    j['diag']
    jac_vals = j['vals']
    j['rhs']
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
