import os
import numpy as np
from model_fn import Model, mesh_prefix
from darts.engines import redirect_darts_output
import shutil
import pickle
import pandas as pd
import matplotlib.pyplot as plt

time_data = pickle.load(open('time_data.pkl', 'rb'))

##############################################################################

td = time_data
for k in td.keys():
    if 'temperature' in k:
        td[k] -= 273.15
        td = td.rename(columns={k: k.replace('(K)', '(degrees)')})
    else: # for rates
        td[k] = np.abs(td[k])

plot_cols = ['BHP', 'temperature', 'water rate']
fig, ax = plt.subplots(1, len(plot_cols), figsize=(12,5))
axx = fig.axes

# plot the defined columns for all wells
for i, col in enumerate(plot_cols):
    y = td.filter(like=col).columns.to_list()
    td.plot(x='Time (years)', y=y, ax=axx[i])
    axx[i].set_ylabel('%s %s'%(col, td.filter(like=col).columns.tolist()[0].split(' ')[-1]))
    l = labels=[lab.split(':')[0].split('(')[0] for lab in axx[i].get_legend_handles_labels()[1]]
    axx[i].legend(l,frameon=False, ncol=2)
    axx[i].tick_params(axis=u'both', which=u'both',length=0)
    for location in ['top','bottom','left','right']:
        axx[i].spines[location].set_linewidth(0)
        axx[i].grid(alpha=0.3)
plt.tight_layout()
plt.show()
