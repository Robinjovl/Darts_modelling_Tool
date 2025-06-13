import pandas as pd
import numpy as np
import re
import h5py
import subprocess
import shutil
import os
import matplotlib
matplotlib.use('pgf')
matplotlib.rc('pgf', texsystem='pdflatex',
               preamble=(
                    r'\usepackage{color}'
                    r'\definecolor{violet}{RGB}{238,130,238}'
                    r'\definecolor{orange}{RGB}{255,165,0}'
               ))
from matplotlib import pyplot as plt
from matplotlib import rcParams
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('legend',fontsize=16)
from PIL import Image

def plot_new_profiles(m):
    props_names = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name
    timesteps, property_array = m.output_properties(output_properties=props_names)

    # folder
    t = round(m.physics.engine.t, 4)
    path = os.path.join(m.output_folder, str(t))
    if not os.path.exists(path):
        os.makedirs(path)

    ls = 18
    x = m.reservoir.discretizer.centroids_all_cells[:, 0]
    for prop, data in property_array.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, data[0, :], color='b', label=prop)
        ax.set_xlabel('distance, m', fontsize=16)
        t_hours = round(m.physics.engine.t * 24, 4)
        ax.text(0.71, 0.85, 'time = ' + str(t_hours) + ' hours', fontsize=16, rotation='horizontal', transform=fig.transFigure)

        # y-axis limits
        if prop == 'p':
            ax.set_ylabel('pressure, bar', fontsize=16)
            ax.set_ylim(m.pressure_init - 0.01, m.pressure_init + 0.1)
        elif prop == 'porosity':
            ax.set_ylabel('porosity', fontsize=16)
            ax.set_ylim(-0.01, 1.01)
        else:
            ax.set_ylabel(prop, fontsize=16)
            # fig.gca().set_ylim(bottom=-0.0001)

        fig.tight_layout()
        fig_name = os.path.join(path, f'{prop}.png')
        fig.savefig(fig_name, dpi=300)
        # plt.show()
        plt.close(fig)

def plot_current_profiles(m, output_folder='./', plot_kinetics=False):
    n_cells = m.reservoir.n
    n_vars = m.physics.nc
    Xm = np.asarray(m.physics.engine.X[:n_cells * n_vars]).reshape(n_cells, n_vars)
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]

    n_plots = 3
    fig, ax = plt.subplots(nrows=n_plots, sharex=True, figsize=(6, 11))

    x = m.reservoir.discretizer.centroids_all_cells[:,0]

    ax[0].plot(x, Xm[:, 0], color='b', label=m.physics.vars[0])
    ax1 = ax[1].twinx()
    colors = ['b', 'r', 'g', 'm', 'cyan']

    for i in [1,4]: # Solid / O
        label = r'$\mathrm{CaCO_3}$(s)' if i == 1 else m.physics.vars[i]
        ax[1].plot(x, Xm[:, i], color=colors[i-1], label=label)
    for i in [2,3]: # Ca / C
        ax1.plot(x, Xm[:, i], color=colors[i-1], label=m.physics.vars[i])
    ax[1].plot(x, 1.0 - np.sum(Xm[:, 2:], axis=1), color=colors[n_vars - 1], label=m.physics.property_containers[0].components_name[-1])
    ax[2].plot(x, poro, color=colors[0], label='porosity')

    t = round(m.physics.engine.t * 24, 4)
    ax[0].text(0.21, 0.9, 'time = ' + str(t) + ' hours',
               fontsize=16, rotation='horizontal', transform=fig.transFigure)
    ax[0].set_ylabel('pressure, bar', fontsize=16)
    ax[n_plots - 1].set_xlabel('distance, m', fontsize=16)
    ax[1].set_ylabel(r'\textcolor{blue}{z$_{CaCO3}$(s)}, \textcolor{magenta}{z$_O$} and \textcolor{cyan}{z$_H$}', fontsize=20)
    ax[2].set_ylabel('porosity', fontsize=16)
    ax1.set_ylabel(r'\textcolor{red}{z$_{Ca}$} and \textcolor{green}{z$_C$}', fontsize=20)

    ax[0].set_ylim(m.pressure_init - 0.01, m.pressure_init + 0.3)
    ax[1].set_ylim(-0.01, 1.01)
    ax[2].set_ylim(-0.01, 1.01)
    ax1.set_ylim(-0.001, 0.01)
    ls = 14
    ax[0].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax[1].legend(loc='upper left', prop={'size': ls}, framealpha=0.9)
    ax[2].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax1.legend(loc='upper right', prop={'size': ls}, framealpha=0.9)

    fig.tight_layout()
    fig_name = os.path.join(output_folder, f'1d_dissolution_time_{round(m.physics.engine.t, 4)}.png')
    fig.savefig(fig_name, dpi=300)
    # plt.show()
    plt.close(fig)

    if plot_kinetics:
        evaluator = m.physics.reservoir_operators[0]
        op_kin_rates = op_vals[:m.reservoir.mesh.n_res_blocks, \
                       evaluator.KIN_OP:evaluator.KIN_OP + n_vars]
        op_sr = op_vals[:m.reservoir.mesh.n_res_blocks, 40]
        op_actHp = op_vals[:m.reservoir.mesh.n_res_blocks, 41]

        ms = 4
        n = 5
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 12))

        kin_rates, SR, actHp = np.zeros(n_cells), np.zeros(n_cells), np.zeros(n_cells)
        for i in range(n_cells):
            _, _, _, rho_phases, kin_state, _ = m.physics.reservoir_operators[0].property.flash_ev.evaluate(Xm[i])
            nu_s = Xm[i, 1]
            nu_a = 1 - nu_s
            rho_s = m.physics.reservoir_operators[0].property.rock_density_ev['Solid_CaCO3'].evaluate(Xm[i, 0]) / m.physics.reservoir_operators[0].property.Mw['Solid_CaCO3']
            rho_a = rho_phases['aq']
            ss = nu_s / rho_s / (nu_a / rho_a + nu_s / rho_s)
            kin_rates[i] = m.physics.reservoir_operators[0].property.kinetic_rate_ev.evaluate(kin_state, ss, rho_s,
                                                              m.physics.reservoir_operators[0].min_z,
                                                              m.physics.reservoir_operators[0].kin_fact)

            SR[i] = kin_state['SR']
            actHp[i] = kin_state['Act(H+)']

        for i in range(n_vars - 1):
            label = 'Operator Rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, op_kin_rates[:, i], color=colors[i], label=label)
            label = 'True rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, m.physics.reservoir_operators[0].input_data.stoich_matrix[i] * kin_rates,
                       color=colors[i], linestyle='--', label=label)
            ax[0].plot(x[::n], m.physics.reservoir_operators[0].input_data.stoich_matrix[i] * kin_rates[::n],
                       color=colors[i], linestyle='None', marker='o', markerfacecolor='none', markersize=ms, label='_nolegend_')
        ax[1].plot(x, op_sr, color='b', label='Operator SR')
        ax[1].plot(x, SR, color='b', linestyle='--', label='True SR')
        ax[1].plot(x[::n], SR[::n], color='b', linestyle='None', marker='o',
                   markerfacecolor='none', markersize=ms, label='_nolegend_')

        ax[2].plot(x, op_actHp, color='r', label=r'Operator $a_{H+}$')
        ax[2].plot(x, actHp, color='r', linestyle='--', label=r'True $a_{H+}$')
        ax[2].plot(x[::n], actHp[::n], color='r', linestyle='None', marker='o',
                   markerfacecolor='none', markersize=ms, label='_nolegend_')
        # mult = SR
        # mult[SR > 100] = 100
        # ax[1].plot(x, 1 - mult, color='orange', linestyle='--', label=r'$1 - \hat{SR}$')

        ax[0].legend(loc='upper right', prop={'size': 12}, framealpha=0.9)
        ax[1].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
        ax[2].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
        ax[len(ax) - 1].set_xlabel('distance, m', fontsize=16)
        ax[0].set_ylabel('Kinetic rate, kmol / day / m3', fontsize=16)
        ax[1].set_ylabel('Saturation Ratio (SR)', fontsize=16)
        ax[2].set_ylabel('H+ ion activity', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f'1d_kin_rate_{t}.png'), dpi=300)
        plt.close(fig)

def plot_profiles(m, output_folder='./', plot_kinetics=False, plot_saturation=True):
    n_cells = m.reservoir.n
    n_vars = m.physics.nc
    n_solid = m.n_solid
    Xm = np.asarray(m.physics.engine.X[:n_cells * n_vars]).reshape(n_cells, n_vars)
    op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
    poro = op_vals[:m.reservoir.mesh.n_res_blocks, m.physics.reservoir_operators[0].PORO_OP]

    # fig = plt.figure(figsize=(16, 6))
    # # Create a 1x3 grid with different column widths (adjust as needed):
    # gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[0.3, 0.4, 0.3])#, wspace=0.2)
    # ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    prop = m.physics.reservoir_operators[0].property
    props_names = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name
    timesteps, property_array = m.output_properties(output_properties=props_names)

    n_plots = 3
    fig, ax = plt.subplots(ncols=n_plots, sharex=True, figsize=(16, 6))

    x = m.reservoir.discretizer.centroids_all_cells[:,0]

    ax[0].plot(x, Xm[:, 0], color='b', label=m.physics.vars[0])
    ax1 = ax[1].twinx()
    colors = {'Solid_CaCO3': 'b', 'Ca': 'r', 'C': 'g', 'O': 'm', 'H': 'cyan', 'Mg': 'orange', 'Solid_CaMg(CO3)2': 'violet'}

    components = m.physics.components
    for comp in ['Solid_CaCO3', 'O']: # Solid / O
        idx = components.index(comp) + 1
        label = m.physics.vars[idx]
        ax[1].plot(x, Xm[:, idx], color=colors[comp], label=label)
    ax[1].plot(x, 1.0 - np.sum(Xm[:, n_solid + 1:], axis=1), color=colors[components[-1]], label=components[-1])

    for comp in ['Ca', 'C']: # Ca / C
        idx = components.index(comp) + 1
        ax1.plot(x, Xm[:, idx], color=colors[comp], label=m.physics.vars[idx])

    y_axis_label1 = r'\textcolor{blue}{z$_{CaCO3(s)}$}, \textcolor{magenta}{z$_O$}, \textcolor{cyan}{z$_H$}'
    y_axis_label11 = r'\textcolor{red}{z$_{Ca}$}, \textcolor{green}{z$_C$}'
    if 'Solid_CaMg(CO3)2' in components:
        idx = components.index('Solid_CaMg(CO3)2') + 1
        label = m.physics.vars[idx]
        ax[1].plot(x, Xm[:, idx], color=colors['Solid_CaMg(CO3)2'], label=label)
        y_axis_label1 += r', \textcolor{violet}{z$_{CaMg(CO3)2(s)}$}'
        idx = components.index('Mg') + 1
        label = m.physics.vars[idx]
        ax1.plot(x, Xm[:, idx], color=colors['Mg'], label=label)
        y_axis_label11 += r', \textcolor{orange}{z$_{Mg}$}'

    ax[2].plot(x, poro, color='b', label='porosity')

    t = round(m.physics.engine.t * 24, 4)
    ax[0].text(0.15, 0.85, 'time = ' + str(t) + ' hours',
               fontsize=16, rotation='horizontal', transform=fig.transFigure)
    ax[0].set_ylabel('pressure, bar', fontsize=16)
    ax[0].set_xlabel('distance, m', fontsize=16)
    ax[1].set_xlabel('distance, m', fontsize=16)
    ax[2].set_xlabel('distance, m', fontsize=16)
    ax[1].set_ylabel(y_axis_label1, fontsize=16)
    ax[2].set_ylabel(r'\textcolor{blue}{porosity}', fontsize=16)
    ax1.set_ylabel(y_axis_label11, fontsize=20)

    ax[0].set_ylim(m.pressure_init - 0.01, m.pressure_init + 0.15)
    ax[1].set_ylim(-0.01, 1.01)
    ax[2].set_ylim(-0.01, 1.01)
    ax1.set_ylim(-0.001, 0.01)
    ls = 14
    ax[0].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
    ax[1].legend(loc='upper left', prop={'size': ls}, framealpha=0.9)
    ax[2].legend(loc='upper left', prop={'size': ls}, framealpha=0.9)
    ax1.legend(loc='upper right', prop={'size': ls}, framealpha=0.9)

    if plot_saturation:
        ax_sat = ax[2].twinx()
        gas_sat = np.zeros(n_cells)
        for i in range(n_cells):
            nu_v, _, _, rho_phases, _, _, _ = prop.flash_ev.evaluate(Xm[i])
            nu_s = Xm[i, 1:1 + n_solid]
            nu_v = nu_v * (1 - nu_s.sum())  # convert to overall molar fraction
            nu_a = 1 - nu_v - nu_s.sum()
            rho_a, rho_v = rho_phases['aq'], rho_phases['gas']
            rho_s = np.array([v.evaluate(Xm[i, 0]) / prop.Mw[k] for k, v in prop.rock_density_ev.items()])
            if nu_v > 0:
                sum = nu_v / rho_v + nu_a / rho_a + (nu_s / rho_s).sum()
                sv = nu_v / rho_v / sum
                # sa = nu_a / rho_a / sum
                # ss = nu_s / rho_s / sum
            else:
                sv = 0
                # sa = nu_a / rho_a / (nu_a / rho_a + nu_s / rho_s)
                # ss = nu_s / rho_s / (nu_a / rho_a + nu_s / rho_s)
            gas_sat[i] = sv

        ax_sat.plot(x, gas_sat, color='r', label='gas saturation')
        ax_sat.set_ylabel(r'\textcolor{red}{gas saturation}', fontsize=16)
        ax_sat.legend(loc='upper right', prop={'size': ls}, framealpha=0.9)

    fig.tight_layout()

    pos0 = ax[0].get_position()  # [x0, y0, width, height]
    pos1 = ax[1].get_position()
    pos2 = ax[2].get_position()
    ax[0].set_position([pos0.x0 + 0.05, pos0.y0, pos0.width, pos0.height])
    fig_name = os.path.join(output_folder, f'1d_dissolution_time_{round(m.physics.engine.t, 4)}.png')
    fig.savefig(fig_name, dpi=300)
    # plt.show()
    plt.close(fig)

    if plot_kinetics:
        evaluator = m.physics.reservoir_operators[0]
        op_kin_rates = op_vals[:m.reservoir.mesh.n_res_blocks, \
                       evaluator.KIN_OP:evaluator.KIN_OP + n_vars]
        op_sr = op_vals[:m.reservoir.mesh.n_res_blocks, 40]
        op_actHp = op_vals[:m.reservoir.mesh.n_res_blocks, 41]

        ms = 4
        n = 5
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 12))

        kin_rates, SR, actHp = np.zeros(n_cells), np.zeros(n_cells), np.zeros(n_cells)
        for i in range(n_cells):
            _, _, _, rho_phases, kin_state, _ = m.physics.reservoir_operators[0].property.flash_ev.evaluate(Xm[i])
            nu_s = Xm[i, 1]
            nu_a = 1 - nu_s
            rho_s = m.physics.reservoir_operators[0].property.rock_density_ev['Solid_CaCO3'].evaluate(Xm[i, 0]) / m.physics.reservoir_operators[0].property.Mw['Solid_CaCO3']
            rho_a = rho_phases['aq']
            ss = nu_s / rho_s / (nu_a / rho_a + nu_s / rho_s)
            kin_rates[i] = m.physics.reservoir_operators[0].property.kinetic_rate_ev.evaluate(kin_state, ss, rho_s,
                                                              m.physics.reservoir_operators[0].min_z,
                                                              m.physics.reservoir_operators[0].kin_fact)

            SR[i] = kin_state['SR']
            actHp[i] = kin_state['Act(H+)']

        for i in range(n_vars - 1):
            label = 'Operator Rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, op_kin_rates[:, i], color=colors[i], label=label)
            label = 'True rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, m.physics.reservoir_operators[0].input_data.stoich_matrix[i] * kin_rates,
                       color=colors[i], linestyle='--', label=label)
            ax[0].plot(x[::n], m.physics.reservoir_operators[0].input_data.stoich_matrix[i] * kin_rates[::n],
                       color=colors[i], linestyle='None', marker='o', markerfacecolor='none', markersize=ms, label='_nolegend_')
        ax[1].plot(x, op_sr, color='b', label='Operator SR')
        ax[1].plot(x, SR, color='b', linestyle='--', label='True SR')
        ax[1].plot(x[::n], SR[::n], color='b', linestyle='None', marker='o',
                   markerfacecolor='none', markersize=ms, label='_nolegend_')

        ax[2].plot(x, op_actHp, color='r', label=r'Operator $a_{H+}$')
        ax[2].plot(x, actHp, color='r', linestyle='--', label=r'True $a_{H+}$')
        ax[2].plot(x[::n], actHp[::n], color='r', linestyle='None', marker='o',
                   markerfacecolor='none', markersize=ms, label='_nolegend_')
        # mult = SR
        # mult[SR > 100] = 100
        # ax[1].plot(x, 1 - mult, color='orange', linestyle='--', label=r'$1 - \hat{SR}$')

        ax[0].legend(loc='upper right', prop={'size': 12}, framealpha=0.9)
        ax[1].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
        ax[2].legend(loc='upper right', prop={'size': ls}, framealpha=0.9)
        ax[len(ax) - 1].set_xlabel('distance, m', fontsize=16)
        ax[0].set_ylabel('Kinetic rate, kmol / day / m3', fontsize=16)
        ax[1].set_ylabel('Saturation Ratio (SR)', fontsize=16)
        ax[2].set_ylabel('H+ ion activity', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f'1d_kin_rate_{t}.png'), dpi=300)
        plt.close(fig)

    return fig_name

def animate_1d(output_folder, fig_paths):
    # Rename them to a zero-padded sequence: frame_0000.png, frame_0001.png, etc.
    # so ffmpeg can pick them up in order.
    for i, old_name in enumerate(fig_paths):
        new_name = os.path.join(output_folder, f"frame_{i:04d}.png")
        if os.path.exists(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)

    # Now call ffmpeg to produce a video "simulation_animation.mp4"
    fps = 1
    ffmpeg_path = r'c:\work\packages\ffmpeg-6.0\bin\ffmpeg.exe'
    output_video = os.path.join(output_folder, "simulation_animation.mp4")
    cmd = [
        ffmpeg_path,
        '-y',
        '-framerate', str(fps),  # frames per second
        '-i', os.path.join(output_folder, 'frame_%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure dimensions are even
        '-c:v', 'libx264',  # H.264 encoding
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-movflags', '+faststart',
        output_video
    ]
    subprocess.run(cmd, check=True)

def write_2d_output_for_paper(paths):
    import pyvista as pv

    m = Model(domain='2D', nx=100)
    m.init()
    nv = m.physics.n_vars
    ncells = m.reservoir.n

    for path in paths:
        # Read the VTK file using pyvista
        mesh = pv.read(path)

        # Check if there is cell data and modify it
        if mesh.cell_data:
            # add hydrogen
            mesh.cell_data['H'] = 1.0 - mesh.cell_data['O'] - mesh.cell_data['Ca'] - mesh.cell_data['C']

            # add porosity
            X = np.asarray(m.physics.engine.X)
            for k, var in enumerate(m.physics.vars):
                X[k:nv*ncells:nv] = mesh.cell_data[var]
            op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
            poro = op_vals[:m.reservoir.mesh.n_res_blocks, 39]
            mesh.cell_data['porosity'] = poro

        # Create a new file name for the output
        output_path = path.replace(".vts", "_modified.vts")

        # Write the modified mesh back to a new VTK file
        mesh.save(output_path)

def plot_max_cfl(paths, labels, nx, linestyle, colors):

    def extract_data(path):
        times = []
        time_steps = []
        cfl_numbers = []

        # Regular expression pattern to match the lines with T, DT, and CFL values
        pattern = re.compile(r"T = ([0-9.e+-]+), DT = ([0-9.e+-]+).*?CFL=([0-9.e+-]+)")

        # Open and process the log file
        with open(path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
                    time_steps.append(float(match.group(2)))
                    cfl_numbers.append(float(match.group(3)))
        return times, time_steps, cfl_numbers

    # colors = ['b', 'r', 'g', 'm', 'cyan']
    lw = 1
    ls = 12
    fs = 16

    fig, ax = plt.subplots(nrows=1, figsize=(8, 6))

    for i, path in enumerate(paths):
        time, steps, cfl = extract_data(path)
        time = np.array(time)
        steps = np.array(steps)
        cfl = np.array(cfl)
        ax.semilogy(time * 24, cfl, color=colors[i], lw=lw, ls=linestyle[i], label=labels[i])

    ax.set_xlabel('time, hours', fontsize=fs)
    ax.set_ylabel('Max CFL', fontsize=fs)
    ax.legend(loc='lower right', prop={'size': ls}, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'cfl_cmp.png', dpi=300)
    # plt.show()
    plt.close(fig)

def animate_2x3_profiles(output_folder,
                         species_keys=None,
                         fps=1,
                         ffmpeg_path='ffmpeg'):
    """
    Combines six per-species PNGs in each time subfolder of output_folder
    into a 2×3 montage per timestep, then uses ffmpeg to create an MP4.

    Parameters
    ----------
    output_folder : str
        Root folder containing subfolders named by time in hours
        (e.g. '0.00', '0.05', ...), each with:
          zH2O.png, zCO2.png, zHCO3-.png,
          zCaHCO3+.png, z(CO2)2.png, zCa+2.png
    species_keys : list[str], optional
        Basenames (without “.png”) in the 2×3 layout order:
          [zH2O,   zCO2,    zHCO3-,
           zCaHCO3+, z(CO2)2, zCa+2]
        Defaults to the above.
    fps : int, default=1
        Frames per second of the output video.
    ffmpeg_path : str, default='ffmpeg'
        Command or full path for ffmpeg.

    Returns
    -------
    str
        Path to the created MP4.
    """
    # default species order
    if species_keys is None:
        species_keys = [
            'zH2O', 'zCO2', 'zHCO3-',
            'zCaHCO3+', 'z(CO2)2', 'zCa+2'
        ]

    # collect & sort only numeric time‑folders
    time_dirs = []
    for d in os.listdir(output_folder):
        full = os.path.join(output_folder, d)
        if not os.path.isdir(full):
            continue
        try:
            float(d)
        except ValueError:
            continue
        time_dirs.append(d)
    time_dirs.sort(key=lambda name: float(name))

    # make a frames dir
    frames_dir = os.path.join(output_folder, 'frames_2x3')

    # iterate timesteps
    if not os.path.exists(frames_dir):
        for idx, t in enumerate(time_dirs):
            src = os.path.join(output_folder, t)
            # load 6 images
            imgs = []
            for key in species_keys:
                fn = os.path.join(src, f"{key}.png")
                if not os.path.exists(fn):
                    raise FileNotFoundError(f"Missing {fn}")
                imgs.append(Image.open(fn))

            # assume identical size for all
            w, h = imgs[0].size
            canvas = Image.new('RGB', (w * 3, h * 2), (255, 255, 255))
            for i, im in enumerate(imgs):
                row, col = divmod(i, 3)
                canvas.paste(im, (col * w, row * h))

            frame = os.path.join(frames_dir, f"frame_{idx:04d}.png")
            canvas.save(frame)
    else:
        print(f"[animate_2x3_profiles] '{frames_dir}' exists — skipping frame generation")

    # build video via ffmpeg
    out_mp4 = os.path.join(output_folder, '2x3_animation.mp4')
    cmd = [
        ffmpeg_path, '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_mp4
    ]
    subprocess.run(cmd, check=True)
    print(f"Animation saved to {out_mp4}")
    return out_mp4

def animate_2x3_profiles_from_sources(
    h5_paths,
    labels,
    species_keys=None,
    output_folder='.',
    frames_dir_name='frames_2x3_sources',
    fps=2,
    ffmpeg_path='ffmpeg',
    nx_fig=16, ny_fig=8, ls=14
):
    """
    Read multiple .h5 result files, plot each species (2×3 grid)
    with one curve per source, animate over time via ffmpeg.

    Parameters
    ----------
    h5_paths : list[str]
      Paths to your .h5 files.
    labels : list[str]
      Legend labels for each source (same length as h5_paths).
    species_keys : list[str], optional
      Which property names to plot, in this 2×3 order:
          [zH2O, zCO2, zHCO3-, zCaHCO3+, z(CO2)2, zCa+2]
    output_folder : str
      Where to write the frames‑folder and final MP4.
    frames_dir_name : str
      Subfolder under output_folder for the PNG frames.
    fps : int
      Frames per second for the movie.
    ffmpeg_path : str
      ffmpeg executable (e.g. 'ffmpeg' or full path).
    """
    import h5py, numpy as np, matplotlib.pyplot as plt, os, subprocess

    # defaults
    if species_keys is None:
        species_keys = [
            'zH2O', 'zCO2', 'zHCO3-',
            'zCaHCO3+', 'z(CO2)2', 'zCa+2'
        ]
    assert len(h5_paths) == len(labels), "h5_paths and labels must match length"

    # load all data first
    all_times = None
    all_props = []     # list of arrays shape (nt, nc, nprops)
    all_prop_names = None
    all_x = []

    for path in h5_paths:
        with h5py.File(path, 'r') as f:
            # read property names and data
            prop_names = f['dynamic/properties_name'].asstr()[...]
            props      = f['dynamic/properties'][:]     # (nt, nc, nprops)

            nc = props.shape[1]
            x = 0.1 * np.arange(nc) / nc + 0.05 / nc
            times = f['dynamic/time'][:] * 24.0

        # store & check consistency
        if all_prop_names is None:
            all_prop_names = prop_names
        else:
            assert np.all(all_prop_names == prop_names), "property names differ between files"

        if all_times is None:
            all_times = times
        else:
            assert np.allclose(all_times, times), "time arrays differ between files"

        all_props.append(props)
        all_x.append(x)   # assume same x for all

    # figure out indices of the species to plot
    idxs = [ list(all_prop_names).index(k) for k in species_keys ]

    # prepare frames folder
    frames_dir = os.path.join(output_folder, frames_dir_name)

    if os.path.exists(frames_dir):
        print(f"[animate] '{frames_dir}' exists, removing it…")
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    make_frames = True

    colors = ['b', 'r', 'g', 'm']

    # build one figure per time step
    nt = len(all_times)
    for ti in range(nt):
        if not make_frames:
            break

        fig, axes = plt.subplots(2, 3, figsize=(nx_fig, ny_fig), sharex=True)
        axes = axes.flatten()

        tval = all_times[ti]
        for si, key in enumerate(species_keys):
            ax = axes[si]
            for k in range(len(labels)):
                label = labels[k]
                props = all_props[k]
                profile = props[ti, :, idxs[si]]
                ax.plot(all_x[k], profile, color=colors[k], label=label)

            ax.set_title(key)
            ax.tick_params(labelsize=12)

            ax.set_ylabel(key, fontsize=14)
            if si >= 3:
                ax.set_xlabel('distance, m', fontsize=14)

            ax.legend(loc='lower right', fontsize=ls, framealpha=0.5)

        # add a suptitle with time
        fig.suptitle(f"time = {tval:.3f} hours", fontsize=16)
        fig.tight_layout(rect=[0,0,1,0.96])

        # save frame
        fname = os.path.join(frames_dir, f"frame_{ti:04d}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)

    # call ffmpeg
    out_mp4 = os.path.join(output_folder, '2x3_comparison.mp4')
    cmd = [
        ffmpeg_path, '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_mp4
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Animation written to", out_mp4)
    return out_mp4


if __name__ == '__main__':
    # output_folder = 'c:\work\packages\open-darts\models\phreeqc_dissolution\data_for_seminar\output_1D_1000'
    ffmpeg_path = r'c:\work\packages\ffmpeg-6.0\bin\ffmpeg.exe'
    # animate_2x3_profiles(
    #     output_folder=output_folder,
    #     fps=2,                   # 2 frames per second
    #     ffmpeg_path=ffmpeg_path     # or full path to your ffmpeg binary
    # )

    # spatial convergence
    # h5_paths = ['.\\data_for_seminar\\output_1D_200\\nx200.h5',
    #             '.\\data_for_seminar\\output_1D_1000\\nx1000.h5',
    #             '.\\data_for_seminar\\output_1D_5000\\nx5000.h5']
    # labels = ['nx=200', 'nx=1000', 'nx=5000']
    # output_folder = '.\\data_for_seminar\\spatial_conv'
    # animate_2x3_profiles_from_sources(h5_paths=h5_paths, labels=labels,
    #                                   output_folder=output_folder, ffmpeg_path=ffmpeg_path,
    #                                   nx_fig=12, ny_fig=8)

    # OBL convergence
    h5_paths = ['.\\data_for_seminar\\output_1D_1000_0\\nx1000.h5',
                '.\\data_for_seminar\\output_1D_1000_1\\nx1000.h5',
                '.\\data_for_seminar\\output_1D_1000_2\\nx1000.h5']
    labels = [r'$\Delta x_{OBL} = \Delta x_{OBL}^0$',
              r'$\Delta x_{OBL} = \Delta x_{OBL}^0 / 5$',
              r'$\Delta x_{OBL} = \Delta x_{OBL}^0 / 25$']
    output_folder = '.\\data_for_seminar\\obl_conv'
    animate_2x3_profiles_from_sources(h5_paths=h5_paths, labels=labels,
                                      output_folder=output_folder, ffmpeg_path=ffmpeg_path,
                                      nx_fig=12, ny_fig=8, ls=12)

