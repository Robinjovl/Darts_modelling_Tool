import pandas as pd
import numpy as np
import re
import h5py
import subprocess
import shutil
import os
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('pgf')
matplotlib.rc('pgf', texsystem='pdflatex',
               preamble=(
                    r'\usepackage{color}'
                    r'\usepackage{amsmath}'
                    r'\definecolor{violet}{RGB}{238,130,238}'
                    r'\definecolor{orange}{RGB}{255,165,0}'
               ))
from matplotlib import pyplot as plt
from matplotlib import rcParams
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('legend',fontsize=16)
from PIL import Image
import glob

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

def plot_new_profiles(m):
    props_names = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name
    _, property_array = m.output.output_properties(output_properties=props_names)

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

    prop = m.physics.reservoir_operators[0].property
    props_names = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name
    _, property_array = m.output.output_properties(output_properties=props_names)

    n_plots = 3
    fig, ax = plt.subplots(ncols=n_plots, sharex=True, figsize=(16, 6))

    x = m.reservoir.discretizer.centroids_all_cells[:,0]

    ax[0].plot(x, Xm[:, 0], color='b', label=m.physics.vars[0])
    ax1 = ax[1].twinx()
    colors = {'Solid_CaCO3': 'b', 'Ca': 'r', 'C': 'g', 'O': 'm', 'H': 'cyan', 'Mg': 'orange',
                'Solid_CaMg(CO3)2': 'violet', 'Solid_MgCO3': 'brown'}

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
    if 'Solid_MgCO3' in components:
        idx = components.index('Solid_MgCO3') + 1
        label = m.physics.vars[idx]
        ax[1].plot(x, Xm[:, idx], color=colors['Solid_MgCO3'], label=label)
        y_axis_label1 += r', \textcolor{brown}{z$_{MgCO3(s)}$}'

    ax[2].plot(x, property_array['porosity'][0], color='b', label='porosity')

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
        ax_sat.plot(x, property_array['satV'][0], color='r', label='gas saturation')
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
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
        ms = 4
        n = 5
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 12))

        mineral = prop.minerals[0]
        op_vals = np.asarray(m.physics.engine.op_vals_arr).reshape(m.reservoir.mesh.n_blocks, m.physics.n_ops)
        evaluator = m.physics.reservoir_operators[0]
        kin_rates_op = op_vals[:m.reservoir.mesh.n_res_blocks, evaluator.KIN_OP:evaluator.KIN_OP + n_vars]
        kin_rates, SR, actHp = np.zeros(n_cells), np.zeros(n_cells), np.zeros(n_cells)

        for i in range(n_cells):
            _, _, _, rho_phases, kin_state, _, _ = m.physics.reservoir_operators[0].property.flash_ev.evaluate(Xm[i])
            nu_s = Xm[i, 1]
            nu_a = 1 - nu_s
            rho_s = m.physics.reservoir_operators[0].property.rock_density_ev[mineral].evaluate(Xm[i, 0]) / m.physics.reservoir_operators[0].property.Mw[mineral]
            rho_a = rho_phases['aq']
            ss = nu_s / rho_s / (nu_a / rho_a + nu_s / rho_s)
            kin_rates[i] = m.physics.reservoir_operators[0].property.kinetic_rate_ev[mineral].evaluate(kin_state, ss, rho_s)
            SR[i] = kin_state['SR_CaCO3']
            actHp[i] = kin_state['Act(H+)']

        for i in range(n_vars - 1):
            label = 'Operator Rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, kin_rates_op[:, i], color=colors[i], label=label)
            label = 'True rate ' + m.physics.property_containers[0].components_name[i]
            ax[0].plot(x, m.physics.reservoir_operators[0].input_data.stoich_matrix[0, i] * kin_rates,
                       color=colors[i], linestyle='--', label=label)
            ax[0].plot(x[::n], m.physics.reservoir_operators[0].input_data.stoich_matrix[0, i] * kin_rates[::n],
                       color=colors[i], linestyle='None', marker='o', markerfacecolor='none', markersize=ms, label='_nolegend_')
        ax[1].plot(x, property_array['SR_CaCO3'][0], color='b', label='Operator SR')
        ax[1].plot(x, SR, color='b', linestyle='--', label='True SR')
        ax[1].plot(x[::n], SR[::n], color='b', linestyle='None', marker='o',
                   markerfacecolor='none', markersize=ms, label='_nolegend_')

        ax[2].plot(x, property_array['Act(H+)'][0], color='r', label=r'Operator $a_{H+}$')
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

def plot_unknowns_from_h5(h5_paths, time_indices, output_folder, fname, h5_labels, n_cols=4, plot_saturation=False):
    """
    Read all .h5 files in `folder_name` and plot profiles analogous to `plot_profiles` for selected timesteps.

    Parameters
    ----------
    folder_name : str
        Path to folder containing .h5 files with groups:
          'dynamic/properties_name' (asstr),
          'dynamic/properties' (nt x nc x nprops),
          'dynamic/time' (nt, ).
    time_indices : list[int]
        List of time-step indices to plot (0-based, up to nt-1).
    output_folder : str, optional
        Where to save output PNGs. Defaults to './plots'.
    domain_length : float, optional
        Total length of the 1D domain for x-axis. Defaults to 1.0.
    plot_saturation : bool, optional
        If True, plot gas saturation on a twin axis of porosity plot.
    """

    # Grid: 2 rows × n_cols columns
    fig = plt.figure(figsize=(18, 7))
    w_ratios = n_cols * [1]
    gs = fig.add_gridspec(2, n_cols, width_ratios=w_ratios, height_ratios=[1, 1],
                          wspace=0.3, hspace=0.15)
    # First column: pressure, spanning both rows
    if plot_saturation:
        ax_p = fig.add_subplot(gs[0, 0])
        ax_satv = fig.add_subplot(gs[1, 0], sharex=ax_p)
    else:
        ax_p = fig.add_subplot(gs[:, 0])
    # Columns 2,3,4: two rows each
    ax_o = fig.add_subplot(gs[0, 1], sharex=ax_p)
    ax_h = fig.add_subplot(gs[0, 2], sharex=ax_p)
    ax_caco3 = fig.add_subplot(gs[0, 3], sharex=ax_p)
    ax_ca = fig.add_subplot(gs[1, 1], sharex=ax_p)
    ax_c = fig.add_subplot(gs[1, 2], sharex=ax_p)
    ax_poro = fig.add_subplot(gs[1, 3], sharex=ax_p)
    if n_cols > 4:
        ax_dol = fig.add_subplot(gs[0, 4], sharex=ax_p)
        ax_mg = fig.add_subplot(gs[1, 4], sharex=ax_p)
    if n_cols > 5:
        ax_mag = fig.add_subplot(gs[0, 5], sharex=ax_p)

    if len(h5_paths) == 1:
        color_var = ['r']
        color_unvar = ['b']
    else:
        color_var = ['b', 'r', 'g', 'm', 'orange', 'c', 'k']
        color_unvar = ['b', 'r', 'g', 'm', 'orange', 'c', 'k']

    linestyles = ['-', '--', ':', '-.']
    for i, h5_path in enumerate(h5_paths):
        with h5py.File(h5_path, 'r') as f:
            prop_names = f['dynamic/properties_name'].asstr()[...]
            props = f['dynamic/properties'][:]       # shape (nt, nc, nprops)
            var_names = f['dynamic/variable_names'].asstr()[...]
            vars = f['dynamic/X'][:]
            times = f['dynamic/time'][:] * 24.0  # convert to hours

        nt, nc, nprops = props.shape
        x = (0.1 * np.arange(nc) / nc + 0.05 / nc) * 1e+3

        for j, ti in enumerate(time_indices):
            if ti < 0 or ti >= nt:
                print(f"Skipping time index {ti}: out of range (0 to {nt-1}) for {fname}")
                continue

            ls = linestyles[j]
            color = 'r'
            ax_p.plot(x, vars[ti, :, np.where(var_names == 'p')[0][0]], linestyle=ls, color=color_var[i], label='pressure')

            ax_o.plot(x, vars[ti, :, np.where(var_names == 'O')[0][0]], linestyle=ls, color=color_var[i], label='O')
            ax_h.plot(x, props[ti, :, np.where(prop_names == 'H')[0][0]], linestyle=ls, color=color_unvar[i], label='H')
            ax_ca.plot(x, vars[ti, :, np.where(var_names == 'Ca')[0][0]], linestyle=ls, color=color_var[i], label='Ca')
            ax_c.plot(x, vars[ti, :, np.where(var_names == 'C')[0][0]], linestyle=ls, color=color_var[i], label='C')

            ax_poro.plot(x, props[ti, :, np.where(prop_names == 'porosity')[0][0]], linestyle=ls, color=color_unvar[i], label='porosity')
            ax_caco3.plot(x, vars[ti, :, np.where(var_names == 'Solid_CaCO3')[0][0]], linestyle=ls, color=color_var[i], label='CaCO3(s)')

            if plot_saturation:
                ax_satv.plot(x, props[ti, :, np.where(prop_names == 'satV')[0][0]], linestyle=ls, color=color_unvar[i], label='vapour saturation')

            if n_cols > 4:
                ax_dol.plot(x, vars[ti, :, np.where(var_names == 'Solid_CaMg(CO3)2')[0][0]], linestyle=ls, color=color, label='CaMg(CO3)2(s)')
                ax_mg.plot(x, vars[ti, :, np.where(var_names == 'Mg')[0][0]], linestyle=ls, color=color, label='Mg')
            if n_cols > 5:
                ax_mag.plot(x, vars[ti, :, np.where(var_names == 'Solid_MgCO3')[0][0]], linestyle=ls, color=color_var[i], label='MgCO3(s)')

    ax_ca.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_o.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_h.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_c.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if n_cols > 4:
        ax_mg.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # hide x-axes for top line of subfigures
    x_axis_to_hide = [ax_o, ax_h, ax_caco3]
    if n_cols > 4: x_axis_to_hide.append(ax_dol)
    if n_cols > 5: x_axis_to_hide.append(ax_mag)
    if plot_saturation:
        x_axis_to_hide.append(ax_p)
    for top_ax in x_axis_to_hide:
        top_ax.tick_params(labelbottom=False)

    fs = 18

    ax_ca.set_xlabel('distance, mm', fontsize=fs)
    ax_c.set_xlabel('distance, mm', fontsize=fs)
    ax_poro.set_xlabel('distance, mm', fontsize=fs)
    if plot_saturation:
        ax_satv.set_ylim(-0.001, 0.251)
        ax_satv.set_xlabel('distance, mm', fontsize=fs)
        ax_satv.set_ylabel('vapour saturation', fontsize=fs)
    else:
        ax_p.set_xlabel('distance, mm', fontsize=fs)

    ax_p.set_ylabel(r'pressure, bar', fontsize=fs)
    ax_o.set_ylabel(r'zO', fontsize=fs)
    ax_h.set_ylabel(r'zH', fontsize=fs)
    ax_caco3.set_ylabel(r'zCaCO3(s)', fontsize=fs)
    ax_ca.set_ylabel(r'zCa', fontsize=fs)
    ax_c.set_ylabel(r'zC', fontsize=fs)
    ax_poro.set_ylabel(r'porosity', fontsize=fs)
    if n_cols > 4:
        ax_mg.set_xlabel('distance, mm', fontsize=fs)
        ax_mg.set_ylabel('zMg', fontsize=fs)
        ax_dol.set_ylabel('zCaMg(CO3)2(s)', fontsize=fs)
    if n_cols > 5:
        ax_mag.set_ylabel('zMgCO3(s)', fontsize=fs)

    # custom legend for time mapping on ax[1]
    legend_font_size = 14
    legend_lines = [Line2D([0], [0], color='k', linestyle=style)
                    for style in linestyles[:len(time_indices)]]
    legend_labels = [f't = {times[ti]:.2f} h' for ti in time_indices]
    ax_p.legend(legend_lines, legend_labels, loc='upper right',
                prop={'size': legend_font_size}, framealpha=0.9)

    if len(h5_paths) > 1 and plot_saturation:
        legend_lines = [Line2D([0], [0], color=color)
                        for color in color_var[:len(h5_paths)]]
        ax_satv.legend(legend_lines, h5_labels, loc='upper right',
                    prop={'size': legend_font_size}, framealpha=0.9)

    #
    # fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)

    outname = os.path.join(output_folder, fname)
    fig.savefig(outname, dpi=300)
    plt.close(fig)

def plot_properties_from_h5(h5_paths, time_indices, output_folder, fname, props_to_plot, h5_labels, nrows=2, ncols=4, nx_fig=18):
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, figsize=(nx_fig, 8))

    linestyles = ['-', '--', ':', '-.']
    colors = ['b', 'r', 'g', 'm', 'orange', 'c', 'k']

    def _xz_variants(label: str):
        variants = [label]
        if label and label[0] in ('x', 'z'):
            swapped = ('z' if label[0] == 'x' else 'x') + label[1:]
            if swapped not in variants:
                variants.append(swapped)
        return variants

    def _special_variant(label: str):
        if label and len(label) > 1 and label[0] in ('x', 'z') and label[1:] == 'CaHCO3+':
            return label[0] + 'Ca(HCO3)+'
        return None

    def _find_property_index(label, prop_names_arr):
        if not label:
            return None
        for candidate in _xz_variants(label):
            idx = np.where(prop_names_arr == candidate)[0]
            if len(idx) > 0:
                return idx[0]

            special_candidate = _special_variant(candidate)
            if special_candidate:
                idx = np.where(prop_names_arr == special_candidate)[0]
                if len(idx) > 0:
                    return idx[0]

            if not candidate.endswith('(aq)'):
                idx = np.where(prop_names_arr == f'{candidate}(aq)')[0]
                if len(idx) > 0:
                    return idx[0]
        return None

    for i, h5_path in enumerate(h5_paths):
        with h5py.File(h5_path, 'r') as f:
            prop_names = f['dynamic/properties_name'].asstr()[...]
            props = f['dynamic/properties'][:]       # shape (nt, nc, nprops)
            var_names = f['dynamic/variable_names'].asstr()[...]
            vars = f['dynamic/X'][:]
            times = f['dynamic/time'][:] * 24.0  # convert to hours

        nt, nc, nprops = props.shape
        x = (0.1 * np.arange(nc) / nc + 0.05 / nc) * 1e+3

        for j, ti in enumerate(time_indices):
            if ti < 0 or ti >= nt:
                print(f"Skipping time index {ti}: out of range (0 to {nt-1}) for {fname}")
                continue

            ls = linestyles[j]

            i_prop = 0
            for k in range(nrows):
                for l in range(ncols):
                    idd = _find_property_index(props_to_plot[i_prop], prop_names)
                    if idd is None:
                        continue
                    ax[k, l].plot(x, props[ti, :, idd], color=colors[i], linestyle=ls, label=props_to_plot[i_prop])
                    i_prop = i_prop + 1

    fs = 18
    i_prop = 0
    for k in range(nrows):
        for l in range(ncols):
            ax[k, l].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax[k, l].set_ylabel(props_to_plot[i_prop], fontsize=fs)
            i_prop += 1

    # hide x-axes for top line of subfigures
    for k in range(ncols):
        ax[0, k].tick_params(labelbottom=False)
        ax[nrows - 1, k].set_xlabel('distance, mm', fontsize=fs)

    # custom legend for time mapping on ax[1]
    legend_font_size = 16
    legend_lines = [Line2D([0], [0], color='k', linestyle=style)
                    for style in linestyles[:len(time_indices)]]
    legend_labels = [f't = {times[ti]:.2f} h' for ti in time_indices]
    ax[0,0].legend(legend_lines, legend_labels, loc='center right',
                prop={'size': legend_font_size}, framealpha=0.9)

    if len(h5_paths) > 1:
        legend_lines = [Line2D([0], [0], color=color)
                        for color in colors[:len(h5_paths)]]
        ax[1,0].legend(legend_lines, h5_labels, loc='upper right',
                    prop={'size': legend_font_size}, framealpha=0.9)

    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.5)

    outname = os.path.join(output_folder, fname)
    fig.savefig(outname, dpi=300)
    plt.close(fig)

def plot_convergence(h5_paths, labels, output_figure, kind, time_index: int = -1, fs: int = 18, ms: int = 7, ls: int = 16):
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                           gridspec_kw={'height_ratios': [5, 1]})
    colors = ['b', 'r', 'g', 'm']
    linestyles = ['-', '-', '-', '-']


    # ax2 = ax.twinx()
    # ax.set_xscale('log')
    # To avoid bar overlap on a log x-axis, use multiplicative offsets around 1.0
    n_series = len(h5_paths)
    if n_series == 1:
        offset_factors = [5.0]
    else:
        # Spread bars gently around the original x (centered at 1.0)
        offset_factors = np.geomspace(0.88, 1.12, n_series).tolist()

    # Store bar handles to build a combined legend that references line color per label
    bar_handles = []
    line_handles = []

    max_dt_all_series = 0.0
    min_error = 1e+6
    for i, paths in enumerate(h5_paths):
        # finest solution
        with h5py.File(paths[-1], 'r') as f:
            finest_vars = f['dynamic/X'][time_index]
            n_cells = finest_vars.shape[0]
            times = f['dynamic/time'][:] * 24.0
            finest_x = 0.1 * np.arange(n_cells) / n_cells + 0.05 / n_cells
            finest_dx = 0.1 / n_cells
            finest_t = times[time_index]

        dx = []
        omega = []
        max_dt = []
        max_cfl = []
        for j in range(len(paths) - 1):
            path = paths[j]
            with h5py.File(path, 'r') as f:
                # read property names and data
                var_names = f['dynamic/variable_names'].asstr()[...]
                vars = f['dynamic/X'][time_index]
                times = f['dynamic/time'][:] * 24.0

            path_well_data = '\\'.join(path.split('\\')[:-1] + ['well_data.h5'])
            with h5py.File(path_well_data, 'r') as f:
                # read property names and data
                dt = np.diff(f['dynamic/time'][14:] * 24.0).max()

            # path_log = '\\'.join(path.split('\\')[:-1] + ['log.txt'])
            # _, _, cfl = extract_data(path_log)
            # max_cfl.append(max(cfl))

            n_cells = vars.shape[0]
            n_vars = vars.shape[1]
            x = 0.1 * np.arange(n_cells) / n_cells + 0.05 / n_cells
            cur_dx = 0.1 / n_cells

            assert(finest_t == times[time_index])

            norm = 0.0
            for k in range(n_vars):
                finest_var_proj = np.interp(x, finest_x, finest_vars[:, k])
                norm += np.sum((vars[:, k] - finest_var_proj) ** 2) / np.sum(finest_var_proj ** 2)
            norm /= n_vars

            if kind == 'spatial':
                dx.append(cur_dx)
            elif kind == 'obl':
                obl_mult = 3
                dx.append(1 / obl_mult ** j)

            omega.append(np.sqrt(norm))
            max_dt.append(dt)

        line, = ax[0].loglog(dx, omega, color=colors[i], linestyle=linestyles[i], markerfacecolor='none', marker='s', markersize=ms, label=labels[i])
        max_dt_all_series = max(max_dt_all_series, np.max(max_dt))
        min_error = min(min_error, np.min(omega))
        # print(max_dt)

        line_handles.append(line)

        # ---- Bars of max_dt (right axis) ----
        # Use multiplicative x-offset so bars don't overlap on a log scale
        x_pos = np.array(dx, dtype=float) * offset_factors[i]
        # Width: a small fraction of x; on log axes this is data-units, so keep modest
        bar_width = np.array(dx, dtype=float) * 0.05

        bars = ax[1].bar(
            x_pos, max_dt,
            width=bar_width,
            align='center',
            alpha=0.6,
            edgecolor='k',
            linewidth=1.0,
            color=colors[i]
        )
        # Keep one handle per label for the legend (use the first bar in the container)
        bar_handles.append(bars.patches[0] if len(bars.patches) else bars)

    if kind == 'obl':
        x_label = r'$\Delta x^{OBL} / \Delta x^{OBL}_0$'
    elif kind == 'spatial':
        x_label = r'$\Delta x$, m'

    ax[1].set_xlabel(x_label, fontsize=fs)
    ax[0].set_ylabel(r'$\left||\boldsymbol{\omega}_h - \boldsymbol{\omega}\right||$', fontsize=fs)
    ax[0].legend(loc='upper left', prop={'size': ls})
    ax[0].grid(True)
    # ax[1].grid(True)

    # ax2.spines['right'].set_visible(False)  # hide the right spine
    # ax2.tick_params(right=False, labelright=False)  # hide right ticks and their labels
    # ax2.get_yaxis().set_visible(False)  # hide the entire right y-axis (safer)
    ax[1].set_ylabel(r'$\max(\Delta t)$, h', fontsize=fs-2)
    # ax[1].set_ylim(top=4 * max_dt_all_series)
    # ax[0].set_ylim(bottom=0.3 * min_error)

    fig.tight_layout()
    fig.savefig(output_figure, dpi=300)
    # plt.show()
    plt.close(fig)

def plot_temporal_convergence_1d(h5_paths, output_figure: str, output_cfl: str, labels: list, time_index: int = -1,
                                 fs: int = 18, ms: int = 7, ls: int = 14):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    colors = ['b', 'r', 'g', 'm', 'orange']
    linestyles = [
        'solid',  # fine
        'dotted',  # fine
        'dashed',  # fine
        'dashdot',  # fine
        (0, (1, 3)),  # replaces 'loosely dotted'
        (0, (3, 1, 1, 1))  # replaces 'densely dotted'
    ]

    for i, paths in enumerate(h5_paths):
        # finest solution
        with h5py.File(paths[-1], 'r') as f:
            finest_vars = f['dynamic/X'][time_index]
            n_cells = finest_vars.shape[0]
            times = f['dynamic/time'][:] * 24.0
            finest_x = 0.1 * np.arange(n_cells) / n_cells + 0.05 / n_cells
            finest_t = times[time_index]

        dx = []
        omega = []
        max_dt = []
        for j in range(len(paths) - 1):
            path = paths[j]
            with h5py.File(path, 'r') as f:
                # read property names and data
                var_names = f['dynamic/variable_names'].asstr()[...]
                vars = f['dynamic/X'][time_index]
                times = f['dynamic/time'][:] * 24.0

            assert (finest_t == times[time_index])

            path_well_data = '\\'.join(path.split('\\')[:-1] + ['well_data.h5'])
            with h5py.File(path_well_data, 'r') as f:
                # read property names and data
                start_time_id = np.diff(f['dynamic/time'][:]).argmin() + 1
                times = f['dynamic/time'][start_time_id:] * 24.0
                dt_max = np.diff(times).max()
                cfl = f['dynamic/CFL_max'][start_time_id:]

            n_cells = vars.shape[0]
            n_vars = vars.shape[1]
            x = 0.1 * np.arange(n_cells) / n_cells + 0.05 / n_cells
            cur_dx = 0.1 / n_cells

            norm = 0.0
            for k in range(n_vars):
                finest_var_proj = np.interp(x, finest_x, finest_vars[:, k])
                norm += np.sum((vars[:, k] - finest_var_proj) ** 2) / np.sum(finest_var_proj ** 2)
            norm /= n_vars

            omega.append(np.sqrt(norm))
            max_dt.append(dt_max)

            ax1.loglog(times, cfl, color=colors[i], linestyle=linestyles[j], label=labels[i] + r': $\max(\Delta t)$=' + str(round(dt_max, 3)) + ' h')

        line, = ax.loglog(max_dt, omega, color=colors[i], linestyle='-', markerfacecolor='none', marker='s', markersize=ms, label=labels[i])

    # fig1
    ax.set_xlabel(r'$\max(\Delta t)$, h', fontsize=fs)
    ax.set_ylabel(r'$\left||\boldsymbol{\omega}_h - \boldsymbol{\omega}\right||$', fontsize=fs)
    ax.legend(loc='upper left', prop={'size': ls})
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_figure, dpi=300)
    plt.close(fig)

    # fig2
    ax1.set_xlabel(r'$t$, h', fontsize=fs)
    ax1.set_ylabel(r'$\max(CFL)$', fontsize=fs)
    ax1.legend(loc='upper left', prop={'size': ls - 6})
    ax1.set_ylim(bottom=1e-2)
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig(output_cfl, dpi=300)
    plt.close(fig1)

def convergence_pictures():
    # spatial convergence
    h5_paths = [['.\\convergence_study\\output_1D_40_calcite_acidic_neutral_carbonate_multilinear_3\\nx40.h5',
                 '.\\convergence_study\\output_1D_200_calcite_acidic_neutral_carbonate_multilinear_3\\nx200.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_3\\nx1000.h5',
                 '.\\convergence_study\\output_1D_5000_calcite_acidic_neutral_carbonate_multilinear_3\\nx5000.h5'],
                ['.\\convergence_study\\output_1D_40_calcite_acidic_neutral_carbonate_multilinear_3_gas\\nx40.h5',
                 '.\\convergence_study\\output_1D_200_calcite_acidic_neutral_carbonate_multilinear_3_gas\\nx200.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_3_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_5000_calcite_acidic_neutral_carbonate_multilinear_3_gas\\nx5000.h5'],
                ['.\\convergence_study\\output_1D_40_calcite_dolomite_acidic_neutral_carbonate_multilinear_3\\nx40.h5',
                 '.\\convergence_study\\output_1D_200_calcite_dolomite_acidic_neutral_carbonate_multilinear_3\\nx200.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_3\\nx1000.h5',
                 '.\\convergence_study\\output_1D_5000_calcite_dolomite_acidic_neutral_carbonate_multilinear_3\\nx5000.h5'],
                ['.\\convergence_study\\output_1D_40_calcite_dolomite_acidic_neutral_carbonate_multilinear_3_gas\\nx40.h5',
                 '.\\convergence_study\\output_1D_200_calcite_dolomite_acidic_neutral_carbonate_multilinear_3_gas\\nx200.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_3_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_5000_calcite_dolomite_acidic_neutral_carbonate_multilinear_3_gas\\nx5000.h5']
                ]
    labels = ['1ph CaCO3', '2ph CaCO3', '1ph CaCO3-CaMg(CO3)2', '2ph CaCO3-CaMg(CO3)2']
    output_figure = '.\\convergence_study\\spatial_conv_new.png'
    # plot_convergence(h5_paths=h5_paths, labels=labels, output_figure=output_figure, kind='spatial')

    # OBL convergence
    h5_paths = [['.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_1\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_3\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_9\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_27\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_81\\nx1000.h5'],
                ['.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_1_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_3_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_9_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_27_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_acidic_neutral_carbonate_multilinear_81_gas\\nx1000.h5'],
                ['.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_1\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_3\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_9\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_27\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_81\\nx1000.h5'],
                ['.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_1_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_3_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_9_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_27_gas\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_acidic_neutral_carbonate_multilinear_81_gas\\nx1000.h5']
                ]
    labels = ['1ph CaCO3', '2ph CaCO3', '1ph CaCO3-CaMg(CO3)2', '2ph CaCO3-CaMg(CO3)2']
    output_figure = '.\\convergence_study\\obl_conv_new.png'
    # plot_convergence(h5_paths=h5_paths, labels=labels, output_figure=output_figure, kind='obl')

    # temporal convergence
    h5_paths = [['.\\convergence_study\\output_1D_1000_calcite_3_0.1_ts_0.01\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_3_0.1_ts_0.001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_3_0.1_ts_0.0001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_3_0.1_ts_1e-05\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_3_0.1_ts_1e-06\\nx1000.h5'],
                ['.\\convergence_study\\output_1D_1000_calcite_dolomite_9_0.1_ts_0.001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_9_0.1_ts_0.0001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_9_0.1_ts_1e-05\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_9_0.1_ts_1e-06\\nx1000.h5'],
                ['.\\convergence_study\\output_1D_1000_calcite_9_1.0_ts_0.01\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_9_1.0_ts_0.001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_9_1.0_ts_0.0001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_9_1.0_ts_1e-05\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_9_1.0_ts_1e-06\\nx1000.h5'],
                ['.\\convergence_study\\output_1D_1000_calcite_dolomite_9_1.0_ts_0.001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_9_1.0_ts_0.0001\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_9_1.0_ts_1e-05\\nx1000.h5',
                 '.\\convergence_study\\output_1D_1000_calcite_dolomite_9_1.0_ts_1e-06\\nx1000.h5'],
                ]
    labels = [r'1ph CaCO3, $\Delta x^{OBL} / \Delta x_0^{OBL} = 3$',
              r'1ph CaCO3-CaMg(CO3)2, $\Delta x^{OBL} / \Delta x_0^{OBL} = 9$',
              r'2ph CaCO3, $\Delta x^{OBL} / \Delta x_0^{OBL} = 9$',
              r'2ph CaCO3-CaMg(CO3)2, $\Delta x^{OBL} / \Delta x_0^{OBL} = 9$',
              ]
    output_figure = '.\\convergence_study\\temp_conv_new.png'
    output_cfl = '.\\convergence_study\\cfl_conv_new.png'
    plot_temporal_convergence_1d(h5_paths=h5_paths, labels=labels, output_figure=output_figure, output_cfl=output_cfl)

def comparison_solvers_databases():
    # 1ph calcite
    output_folder = 'cmp_solvers'

    # 1ph calcite
    h5_paths = [os.path.join('cmp_solvers/output_1D_1000_calcite_9_0.1_ts_0.0001_phreeqc_phreeqc', 'nx1000.h5'),
                os.path.join('cmp_solvers/output_1D_1000_calcite_9_0.1_ts_0.0001_phreeqc_pitzer', 'nx1000.h5'),
                os.path.join('cmp_solvers/output_1D_1000_calcite_9_0.1_ts_0.0001_reaktoro_phreeqc', 'nx1000.h5'),
                os.path.join('cmp_solvers/output_1D_1000_calcite_9_0.1_ts_0.0001_reaktoro_supcrtbl', 'nx1000.h5')]
    h5_labels = ['phreeqc: phreeqc', 'phreeqc: pitzer', 'reaktoro: phreeqc', 'reaktoro: supcrtbl']
    time_indices = [1, 2, 3, 13]
    plot_unknowns_from_h5(h5_paths=h5_paths, output_folder=output_folder,
                          fname='vars_calcite_1ph_1D.png', time_indices=time_indices, h5_labels=h5_labels,
                          plot_saturation=True)
    # props = ['xH2O', 'xCO2', 'xHCO3-', 'xCaHCO3+',
    #          'x(CO2)2', 'xCa+2', 'SR_CaCO3', 'rate_CaCO3', 'xOH-', 'xH+',
    #          'xCH4', 'xCO3-2', 'xCaCO3', 'xCaOH+', 'xH2', 'xO2']
    props = ['xH2O', 'xCO2', 'xHCO3-', 'xCaHCO3+', 'x(CO2)2',
             'xCa+2', 'SR_CaCO3', 'rate_CaCO3']
    plot_properties_from_h5(h5_paths=h5_paths, time_indices=time_indices, output_folder=output_folder,
                            h5_labels=h5_labels, fname='props_calcite_1ph_1D.png', props_to_plot=props, nrows=2, ncols=4)

    # 2ph calcite
    h5_paths = [os.path.join('cmp_solvers/output_1D_1000_calcite_9_1.0_ts_0.0001_phreeqc_phreeqc', 'nx1000.h5'),
                os.path.join('cmp_solvers/output_1D_1000_calcite_9_1.0_ts_0.0001_phreeqc_pitzer', 'nx1000.h5'),
                #os.path.join('cmp_solvers/output_1D_1000_calcite_9_1.0_ts_0.0001_phreeqc_supcrtbl', 'nx1000.h5'),
                os.path.join('cmp_solvers/output_1D_1000_calcite_9_1.0_ts_0.0001_reaktoro_phreeqc', 'nx1000.h5'),
                os.path.join('cmp_solvers/output_1D_1000_calcite_9_1.0_ts_5e-05_reaktoro_supcrtbl', 'nx1000.h5')
                ]
    h5_labels = ['phreeqc: phreeqc', 'phreeqc: pitzer', 'reaktoro: phreeqc', 'reaktoro: supcrtbl']
    time_indices = [1, 2, 3, 13]
    plot_unknowns_from_h5(h5_paths=h5_paths, output_folder=output_folder,
                          fname='vars_calcite_2ph_1D.png', time_indices=time_indices, h5_labels=h5_labels,
                          plot_saturation=True)
    props = ['xH2O', 'xCO2', 'xHCO3-', 'xCaHCO3+', 'x(CO2)2',
            'xCa+2', 'xCO2(g)', 'xH2O(g)', 'SR_CaCO3', 'rate_CaCO3']
    plot_properties_from_h5(h5_paths=h5_paths, time_indices=time_indices, output_folder=output_folder, h5_labels=h5_labels,
                            fname='props_calcite_2ph_1D.png', props_to_plot=props, nrows=2, ncols=5)


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
    # animate_2x3_profiles_from_sources(h5_paths=h5_paths, labels=labels,
    #                                   output_folder=output_folder, ffmpeg_path=ffmpeg_path,
    #                                   nx_fig=12, ny_fig=8, ls=12)

    # 1ph calcite
    output_folder = 'output_1D_200_calcite_acidic_neutral_carbonate_multilinear_9_1ph_gas_spec'
    h5_paths = [os.path.join(output_folder, 'nx200.h5')]
    time_indices = [0, 1, 2, 10]
    plot_unknowns_from_h5(h5_paths=h5_paths, output_folder=output_folder,
                            fname='vars_calcite_1ph_1D.png', time_indices=time_indices, plot_saturation=True)
    props = ['zH2O', 'zCO2', 'zHCO3-', 'zCaHCO3+',
             'z(CO2)2', 'zCa+2', 'SR_CaCO3', 'rate_CaCO3', 'zOH-', 'zH+',
             'zCH4', 'zCO3-2', 'zCaCO3', 'zCaOH+', 'zH2', 'zO2']
    plot_properties_from_h5(h5_paths=h5_paths, time_indices=time_indices, output_folder=output_folder,
                            fname='props_calcite_1ph_1D.png', props_to_plot=props, nrows=2, ncols=8)
    # 2ph calcite
    output_folder = 'output_1D_200_calcite_acidic_neutral_carbonate_multilinear_9_2ph_gas_spec'
    h5_paths = [os.path.join(output_folder, 'nx200.h5')]
    time_indices = [0, 1, 2, 10]
    plot_unknowns_from_h5(h5_paths=h5_paths, output_folder=output_folder,
                            fname='vars_calcite_2ph_1D.png', time_indices=time_indices, plot_saturation=True)
    # props = ['zH2O', 'zCO2', 'zHCO3-', 'zCaHCO3+',
    #          'z(CO2)2', 'zCa+2', 'zOH-', 'zH+',
    #          'zCH4', 'zCO3-2', 'zCaCO3', 'zCaOH+', 'zH2', 'zO2', 'zCO2(g)', 'zH2O(g)']
    props = ['zH2O', 'zCO2', 'zHCO3-', 'zCaHCO3+', 'z(CO2)2',
             'zCa+2', 'zCO2(g)', 'zH2O(g)', 'SR_CaCO3', 'rate_CaCO3']
    plot_properties_from_h5(h5_paths=h5_paths, time_indices=time_indices, output_folder=output_folder,
                            fname='props_calcite_2ph_1D.png', props_to_plot=props, nrows=2, ncols=5)
    # 1ph dolomite
    output_folder = 'output_1D_200_calcite_dolomite_magnesite_acidic_neutral_carbonate_multilinear_3_1ph_1'
    h5_paths = [os.path.join(output_folder, 'nx200.h5')]
    time_indices = [0, 1, 2, 10]
    plot_unknowns_from_h5(h5_paths=h5_paths, output_folder=output_folder,
                            fname='vars_calcite_dolomite_magnesite_1ph_1D.png', time_indices=time_indices, n_cols=6, plot_saturation=True)
    props = ['zH2O', 'zCO2', 'zHCO3-', 'zCaHCO3+',
             'z(CO2)2', 'zCa+2', 'SR_CaCO3', 'rate_CaCO3', 'zOH-', 'zH+',
             'zCH4', 'zCO3-2', 'zCaCO3', 'zCaOH+', 'zH2', 'zO2',
             'zMgCO3', 'zMg+2', 'zMgOH+', 'zMgHCO3+', 'SR_CaMg(CO3)2', 'rate_CaMg(CO3)2',
             'SR_MgCO3', 'rate_MgCO3']
    plot_properties_from_h5(h5_paths=h5_paths, time_indices=time_indices, output_folder=output_folder,
                            fname='props_calcite_dolomite_magnesite_1ph_1D.png', props_to_plot=props, nrows=2, ncols=12, nx_fig=34)
    # 2ph dolomite
    output_folder = 'output_1D_200_calcite_acidic_neutral_carbonate_multilinear_9_2ph_gas_spec'
    h5_paths = [os.path.join(output_folder, 'nx200.h5')]
    time_indices = [0, 1, 2, 10]
    plot_unknowns_from_h5(h5_paths=h5_paths, output_folder=output_folder,
                            fname='vars_calcite_dolomite_2ph_1D.png', time_indices=time_indices, plot_saturation=True)
    # props = ['zH2O', 'zCO2', 'zHCO3-', 'zCaHCO3+',
    #          'z(CO2)2', 'zCa+2', 'zOH-', 'zH+',
    #          'zCH4', 'zCO3-2', 'zCaCO3', 'zCaOH+', 'zH2', 'zO2', 'zCO2(g)', 'zH2O(g)']
    props = ['zH2O', 'zCO2', 'zHCO3-', 'zCaHCO3+', 'z(CO2)2',
             'zCa+2', 'zCO2(g)', 'zH2O(g)', 'SR_CaCO3', 'rate_CaCO3']
    plot_properties_from_h5(h5_paths=h5_paths, time_indices=time_indices, output_folder=output_folder,
                            fname='props_calcite_dolomite_2ph_1D.png', props_to_plot=props, nrows=2, ncols=5)

    # convergence_pictures()
    # comparison_solvers_databases()
