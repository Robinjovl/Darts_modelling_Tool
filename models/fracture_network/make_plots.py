import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def plot_wells(folder_name, axx, plot_cols):
    linestyle = 'dashed' if '_prev' in folder_name else 'solid'

    pkl_fname = os.path.join(folder_name, 'well_time_data.pkl')
    time_data = pickle.load(open(pkl_fname, 'rb'))

    td = time_data

    for k in td.keys():
        if 'BHT' in k:
            td[k.replace('BHT', 'BHT[degrees]')] = td[k] - 273.15
        else: # for rates
            td[k] = np.abs(td[k])

    # plot the defined columns for all wells
    for i, col in enumerate(plot_cols):
        y = td.filter(like=col).columns.to_list()
        td.plot(x='Time[years]', y=y, ax=axx[i], linestyle=linestyle)
        axx[i].set_ylabel('%s %s'%(col, td.filter(like=col).columns.tolist()[0].split(' ')[-1]))
        l = labels=[lab.split(':')[0].split('(')[0] for lab in axx[i].get_legend_handles_labels()[1]]
        axx[i].legend(l,frameon=False, ncol=2)
        axx[i].tick_params(axis=u'both', which=u'both',length=0)
        for location in ['top','bottom','left','right']:
            axx[i].spines[location].set_linewidth(0)
            axx[i].grid(alpha=0.3)
    plt.tight_layout()

##############################################################

folder_name = 'sol_case_5'

plot_cols = ['BHP', 'BHT[degrees]', 'volumetric_rate_water_at_wh']
for pc in plot_cols:
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    axx = fig.axes
    plot_wells(folder_name, axx, [pc])
    plot_wells(folder_name + '_prev', axx, [pc])
    plt.savefig('diff' + pc + '.png')
    plt.close()
