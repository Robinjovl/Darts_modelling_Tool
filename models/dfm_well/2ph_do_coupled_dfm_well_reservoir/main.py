import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from model import Model
from darts.engines import value_vector, redirect_darts_output


redirect_darts_output('run.log')
n = Model()
n.init()
n.set_output()

n.run(3000)

Xn = np.array(n.physics.engine.X, copy=False)
n_vars = n.physics.n_vars
Xn_reshaped = Xn.reshape(n.reservoir.mesh.n_blocks, n_vars)
np.savetxt('Xn.txt', Xn_reshaped, fmt='%.4f')
