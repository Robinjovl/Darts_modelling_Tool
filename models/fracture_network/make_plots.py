import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def plot_well_data_column(folder_name, axx, plot_cols, plot_cols_hist):
    linestyle = 'dashed' if '_prev' in folder_name else 'solid'

    pkl_fname = os.path.join(folder_name, 'well_time_data.pkl')
    time_data = pickle.load(open(pkl_fname, 'rb'))

    td = time_data
    print(td.keys())

    for k in td.keys():
        if 'BHT' in k:
            td[k.replace('BHT', 'BHT[degrees]')] = td[k] - 273.15
        else: # for rates
            td[k] = np.abs(td[k])

    # plot the defined columns for all wells
    for i, col in enumerate(plot_cols):
        y = td.filter(like=col).columns.to_list()
        y_hist = td.filter(like=plot_cols_hist[i]).columns.to_list()

        td.plot(x='Time(years)', y=y, ax=axx[i], linestyle=linestyle)
        if len(y_hist):
            td.plot(x='Time(years)', y=y_hist, ax=axx[i], linestyle='--', alpha=0.75)

        axx[i].set_ylabel('%s %s'%(col, td.filter(like=col).columns.tolist()[0].split(' ')[-1]))
        l = labels=[lab.split(':')[0].split('(')[0] for lab in axx[i].get_legend_handles_labels()[1]]
        axx[i].legend(l,frameon=False, ncol=2)
        axx[i].tick_params(axis=u'both', which=u'both',length=0)
        for location in ['top','bottom','left','right']:
            axx[i].spines[location].set_linewidth(0)
            axx[i].grid(alpha=0.3)
    plt.tight_layout()



def plot_well_results(folder_name):
    #plot_cols = ['BHP', 'BHT[degrees]', 'volumetric_rate_water_at_wh'] # new time_data columns
    plot_cols = ['BHP', 'temperature', 'water rate'] # old time_data columns
    plot_cols_hist = ['BHP_hist', 'BHT_hist', 'rate_hist']

    for pc,pch in zip(plot_cols, plot_cols_hist):
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        axx = fig.axes
        plot_well_data_column(folder_name, axx, [pc], [pch])
        #plot_well_data_column(folder_name + '_prev', axx, [pc], [pch])
        plt.savefig(os.path.join(folder_name, 'figures', pc + '.png'))
        plt.close()

##############################################################
if __name__ == '__main__':
    folder_name = 'sol_case_1_well_rates'
    plot_well_results(folder_name)
