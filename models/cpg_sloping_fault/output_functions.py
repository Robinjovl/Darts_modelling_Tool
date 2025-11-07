import os
from darts.tools.plot_darts import *

def plot_results(wells, well_is_inj, time_data_list, time_data_report_list, label_list, physics_type, out_dir, case):
    plt.rc('font', size=12)

    for well_name in wells:
        if well_is_inj(well_name):
            continue
        if physics_type == 'geothermal':
            ax = None
            for time_data_report, label in zip(time_data_report_list, label_list):
                ax = plot_temp_darts(well_name, time_data_report, ax=ax)#, label=label)
            ax.set(xlabel="Days", ylabel="temperature [degrees]")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'well_temperature_' + well_name + '_' + case + '.png'))
            plt.close()

            # use time_data here as we are going to compute a cumulative plot
            ax = None
            for time_data, label in zip(time_data_list, label_list):
                ax = plot_extracted_energy_darts(time_data, ax=ax)#, label=label)
            ax.set(xlabel="Days", ylabel="energy [PJ]")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'energy_extracted_' + well_name + '_' + case + '.png'))
            plt.close()
        elif physics_type == 'deadoil':
            # rate plotting
            # TODO need to get proper volumetric rates to compute the watercut
            ax = None
            for time_data_report, label in zip(time_data_report_list, label_list):
                ax = plot_total_prod_oil_rate_darts(time_data_report, ax=ax)#, label=label)
            ax.set(xlabel="Days", ylabel="Total produced oil rate, kmol/day")
            plt.savefig(os.path.join(out_dir, 'production_oil_rate_' + well_name + '_' + case + '.png'), )
            plt.close()

    rate_units = 'm3/day' if physics_type == 'geothermal' else 'kmol/day'

    # common plots for both physics
    ax = None
    for time_data_report, label in zip(time_data_report_list, label_list):
        ax = plot_total_inj_water_rate_darts(time_data_report, ax=ax)#, label=label)
    ax.set(xlabel="Days", ylabel="Total injected water rate, " + rate_units)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'injection_water_rate_' + case + '.png'))
    plt.close()

    ax = None
    for time_data_report, label in zip(time_data_report_list, label_list):
        ax = plot_total_prod_water_rate_darts(time_data_report, ax=ax)#, label=label)
    ax.set(xlabel="Days", ylabel="Total produced water rate, " + rate_units)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'production_water_rate_' + case + '.png'))
    plt.close()

    for well_name in wells:
        ax = None
        for time_data_report, label in zip(time_data_report_list, label_list):
            ax = plot_bhp_darts(well_name, time_data_report, ax=ax)#, label=label)
        ax.set(xlabel="Days", ylabel="BHP [bar]")
        plt.savefig(os.path.join(out_dir, 'well_' + well_name + '_bhp_' + case + '.png'))
        plt.tight_layout()
        plt.close()


def output_vtk(out_dir, m, save_property_array = False):
    # post-processing: read h5 file and write vtk with properties
    print('Post processing properties and vtk output...')
    if "geothermal" in m.physics_type:
        output_properties_main = m.physics.vars + ["temperature"] # main variables + temperature
    else:
        output_properties_main = m.physics.vars # only main variables
    output_properties_full = output_properties_main + m.output.properties  # additional properties (might take some time to compute)
    m.reservoir.create_vtk_wells(output_directory=out_dir)
    n_timesteps = len(m.idata.sim.time_steps)
    for ith_step in range(n_timesteps + 1):
        # compute additional properties only for the first and for the last timestep:
        output_properties = output_properties_full if ith_step in [0, n_timesteps] else output_properties_main
        # print('timestep', ith_step, 'output_properties:', output_properties)
        timesteps, property_array = m.output.output_properties(output_properties=output_properties, timestep=ith_step,
                                                               engine=False)
        if ith_step == 0:
            centers_x, centers_y, centers_z = m.reservoir.get_centers()
            property_array.update({'centers_x': centers_x.reshape(1, -1), 'centers_y': centers_y.reshape(1, -1),
                                   'centers_z': centers_z.reshape(1, -1)})

        if save_property_array:
            m.output.save_property_array(timesteps, property_array, f'property_array_ts{ith_step}.h5')

        m.output.output_to_vtk(output_data=[timesteps, property_array], ith_step=ith_step)

    m.reservoir.centers_to_vtk(os.path.join(out_dir, 'vtk_files'))


def add_columns_time_data(time_data):
    time_data['Time (years)'] = time_data['time'] / 365.25  # extra column with time in years
    for k in time_data.keys():
        # extra column with temperature in celsius
        if 'BHT' in k:
            time_data[k.replace('K', 'degrees')] = time_data[k] - 273.15
            # time_data.drop(columns=k, inplace=True)


def output_time_data(out_dir, m, case, plot_all=True):
    td = m.output.store_well_time_data()
    time_data = pd.DataFrame.from_dict(td)
    add_columns_time_data(time_data)
    time_data.to_pickle(os.path.join(out_dir, 'well_time_data.pkl'))

    # COMPUTE TIME DATA AT FIXED REPORTING STEPS
    if 1:
        mask = (time_data['time'] % 365.25 == 0)
        time_data_report = time_data.loc[mask].copy()
    else:
        time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)
    # add_columns_time_data(time_data_report)
    
    ## filter time_data_report and write to xlsx
    ## list the column names that should be removed
    # press_gridcells = time_data_report.filter(like='reservoir').columns.tolist()
    # chem_cols = time_data_report.filter(like='Kmol').columns.tolist()
    ## remove columns from data
    # time_data_report.drop(columns=press_gridcells + chem_cols, inplace=True)
    
    # add time in years
    time_data_report['Time (years)'] = time_data_report['time'] / 365.25
    time_data_report.to_pickle(os.path.join(out_dir, 'time_data_report.pkl'))
    
    # write both DataFrames to one Excel file
    excel_path = os.path.join(out_dir, 'time_data.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        time_data.to_excel(writer, sheet_name='time_data', index=False)
        time_data_report.to_excel(writer, sheet_name='time_data_report', index=False)

    if plot_all:
        
        # m.output.plot_well_time_data(types_of_well_rates = ["phase_volumetric_rates"]) # use this function instead if you want plots of all the perforations
        plot_well_time_data_2(m, time_data)
        
        # plot_results(wells=m.idata.well_data.wells.keys(), well_is_inj=m.idata.well_is_inj,
        #              time_data_list=[time_data], time_data_report_list=[time_data_report], label_list=[None],
        #              physics_type=m.physics_type, out_dir=out_dir, case=case)

def plot_well_time_data_2(m, time_data_df):
        """
        Make plots out of the time data dataframe.
        """

        m.output.well_plots_dir = os.path.join(m.output_folder, 'figures/well_time_plots')
        # if os.path.exists(m.output.well_plots_dir):
        #     shutil.rmtree(m.output.well_plots_dir)
        os.makedirs(m.output.well_plots_dir, exist_ok=True)

        rate_list = {'volumetric_rate': ' [m3/day]',
                     'mass_rate': ' [kg/day]',
                     'molar_rate': ' [kmol/day]',
                     'advective_heat': ' [kJ]'}
        type_list = ['at_wh'] #, 'by_sum_perfs']

        # for the phases
        for well in m.output.reservoir.wells:
            for phase in m.output.physics.phases:
                for rate in rate_list:
                    y = [f'well_{well.name}_{rate}_{phase}_{type}' for type in type_list]
                    y_valid = [col for col in y if col in time_data_df.columns]
                    if y_valid:
                        time_data_df.plot(x='time', y=y_valid, xlabel = 'time [days]', ylabel=rate + rate_list[rate], title=well.name)\
                            .get_figure().savefig(os.path.join(m.output.well_plots_dir, f'{y_valid[0]}.png'), dpi=100, bbox_inches='tight')
        plt.close()

        # for the components
        for well in m.output.reservoir.wells:
            for component in m.output.physics.components:
                for rate in rate_list:
                    y = [f'well_{well.name}_{rate}_{component}_{type}' for type in type_list]
                    y_valid = [col for col in y if col in time_data_df.columns]
                    if y_valid:
                        time_data_df.plot(x='time', y=y_valid, xlabel = 'time [days]', ylabel=rate + rate_list[rate], title=well.name)\
                            .get_figure().savefig(os.path.join(m.output.well_plots_dir, f'{y_valid[0]}.png'), dpi=100, bbox_inches='tight')
        plt.close()

        # BHP and BHT
        bottom_hole_list = {'BHP': 'Bars', 'BHT': 'C'}
        for well in m.output.reservoir.wells:
            for i in bottom_hole_list.keys():
                y = [f'well_{well.name}_{i}']
                y_valid = [col for col in y if col in time_data_df.columns]
                if y_valid:
                    time_data_df.plot(x='time', y=y_valid, xlabel='time [days]', ylabel=bottom_hole_list[i], title=well.name)\
                        .get_figure().savefig(os.path.join(m.output.well_plots_dir, f'{y_valid[0]}.png'), dpi=100, bbox_inches='tight')
        plt.close()

