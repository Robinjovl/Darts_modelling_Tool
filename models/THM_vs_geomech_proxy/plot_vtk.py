try:
    import pyvista as pv
except ImportError:
    pv = None
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import MaxNLocator
import os
import numpy as np

if pv is not None:
    pv.global_theme.jupyter_backend = 'static' # do not print Widget(...) output messages - they appear in case of pyvista[jupyter] is installed

def plot_slice_matplotlib(slice_plane, arr_name, tensor, component_index, scale,
                          arr_name_plot, contour, rsv_top, rsv_bottom,
                          well_markers, plot_points_xy,
                          xmin_blk, xmax_blk, zmin_blk, zmax_blk, out_png):
    '''
    Optional matplotlib backend for the XZ slice plot (selected via use_mtri=True).
    Gives a centered, evenly-labeled X/Z view with visible 'X, m.'/'Z, m.' titles, which the
    pyvista view_xz camera + show_bounds do not.

    Drawbacks vs the default pyvista rendering:
    - Convex-hull fill: mtri.Triangulation(xs, zs) with no explicit triangles triangulates the
      CONVEX HULL of the points, so non-convex / holey domains (faults, fractures, cut-outs,
      partial domains like case_1) get bridged with spurious triangles and colored where there
      is actually no mesh. pyvista draws only real cells and never invents geometry.
    - Smoothing of discontinuities: the data is cell data; cell_data_to_point_data() averages
      onto points and gouraud shading interpolates across triangles, so sharp jumps (perm,
      material / reservoir boundaries, stress steps) get blurred instead of staying piecewise
      constant per cell.
    - Cell geometry is lost: the re-triangulation does not follow the real cell edges/shapes.
    - 2D-slice only and runs a Delaunay triangulation per array / timestep (extra cost;
      degenerate / coincident points after slicing can make it fail).
    '''
    # interpolate the (cell) scalar to the slice points so tripcolor can render it
    slice_points = slice_plane.cell_data_to_point_data()
    xs = slice_points.points[:, 0]
    zs = slice_points.points[:, 2]
    if tensor:
        vals = np.asarray(slice_points[arr_name])[:, component_index] * scale
    else:
        vals = np.asarray(slice_points[arr_name]) * scale

    # use a single color if values are almost the same everywhere
    plot_rel_diff_threshold = 0.001
    if np.fabs(vals.max() - vals.min()) < plot_rel_diff_threshold:
        vals = np.full_like(vals, vals.min())

    triang = mtri.Triangulation(xs, zs)
    fig, ax = plt.subplots(figsize=(10, 6))
    tpc = ax.tripcolor(triang, vals, shading='gouraud', cmap='viridis')
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.02)
    if contour:
        ax.tricontour(triang, vals, levels=20, colors='black', linewidths=0.5)

    # horizontal reference lines at reservoir top and bottom
    for z_ref in [rsv_top, rsv_bottom]:
        ax.axhline(z_ref, color='white', linewidth=1.0)
    # vertical lines at the wells (prod=red, inj=cyan); top a bit above the visualized block
    z_well_top = zmin_blk - 0.1 * (zmax_blk - zmin_blk)
    for x_well, clr in well_markers:
        ax.vlines(x_well, z_well_top, rsv_bottom, color=clr, linewidth=2)
    # black reference lines at idata.other.points_xy (if set), spanning the full depth
    for p in plot_points_xy:  # [x, y, label]
        ax.vlines(p[0], zmin_blk, zmax_blk, color='black', linewidth=1)

    ax.set_xlim(xmin_blk, xmax_blk)
    ax.set_ylim(zmax_blk, zmin_blk)  # depth increases downward
    ax.set_xlabel('X, m.', fontsize=14)
    ax.set_ylabel('Z, m.', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(arr_name_plot, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def plot_vtk_pyvista(output_dir, idata, contour=False, tstep_to_plot=-1, use_mesh_bounds=False,
                     plot_contours=False, use_mtri=False):
    '''
    Plot VTK results using PyVista.
    saves 2D plots - xz slice - of specified arrays (vertic displ and stress) from the last timestep.

    idata : InputData
        reservoir/well geometry (top/bottom depths, well X positions, plot window,
        reference points) is taken from idata.other so the plots adapt to the actual case.
    use_mesh_bounds : bool
        if True, the X plot window spans the actual mesh extent instead of the
        rsv_xy-based window (5 * rsv_xy). Z always spans the full mesh depth.
    plot_contours : bool
        if True, also produce the matplotlib contour plot (resamples the slice onto a
        uniform grid - skipped entirely when False to avoid the cost).
    use_mtri : bool
        if True, render the slice with matplotlib/tripcolor (plot_slice_matplotlib) instead of
        the default pyvista rendering. Gives centered axes with proper titles, but see the
        drawbacks documented on plot_slice_matplotlib.
    '''

    if pv is None:
        print('pyvista is not installed, skipping VTK plotting')
        return

    if 'sawcut' in output_dir or '2rocks' in output_dir: # contours help to see that u_z is the same along X-axes in the inclined hex mesh
        contour = True

    #filename = os.path.join(output_dir, 'vtk', 'solution.pvd')
    #output_dir_plots = os.path.join(os.path.dirname(os.path.dirname(filename)), 'plots')
    filename = os.path.join(output_dir, 'solution.pvd')
    output_dir_plots = os.path.join(os.path.dirname(filename), 'plots_timestep_' + str(tstep_to_plot))
    os.makedirs(output_dir_plots, exist_ok=True)

    # Get reader and check available timesteps
    reader = pv.get_reader(filename)
    t_steps = np.asarray(reader.time_values)
    print("Available timesteps (days):", t_steps[:5], '...', t_steps[-5:])
    t_steps_years = t_steps / 365.25
    print("Available timesteps (years):", t_steps_years[:5], '...', t_steps_years[-5:])

    # Load the data for the asked timestep
    reader.set_active_time_value(reader.time_values[tstep_to_plot])
    mesh = reader.read()

    # Check if mesh is MultiBlock
    if isinstance(mesh, pv.MultiBlock):
        print(f"MultiBlock with {len(mesh)} blocks")
        # Access the first block
        block = mesh[0]
    else:
        block = mesh

    print("Available cell data keys:", list(block.cell_data.keys()))
    print("Available point data keys:", list(block.point_data.keys()))

    # Define plot configurations: (array name, is tensor?, array name for plot (filename), add contour?)
    plot_config_list = []
    component_index = None

    arr_name = "ux"; tensor = False; arr_name_plot = 'u_x,m'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = "uy"; tensor = False; arr_name_plot = 'u_y,m'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = "uz"; tensor = False; arr_name_plot = 'u_z,m'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = "temperature"; tensor = False; arr_name_plot = 'temperature,K'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = "pressure"; tensor = False; arr_name_plot = 'pressure,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = "delta_temperature"; tensor = False; arr_name_plot = 'delta_temperature,K'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = "delta_pressure"; tensor = False; arr_name_plot = 'delta_pressure,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = "viscosity"; tensor = False; component_index = None; arr_name_plot = 'viscosity,cP'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'perm'; tensor = True; component_index = 0; arr_name_plot = 'perm_XX,mD'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    #arr_name = 'perm'; tensor = True; component_index = 4; arr_name_plot = 'perm_YY,mD'; scale = 1.0
    #plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'perm'; tensor = True; component_index = 8; arr_name_plot = 'perm_ZZ,mD'; scale = 1.0
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'delta_eff_stress'; tensor = True; component_index = 2; arr_name_plot = 'delta_eff_stress_ZZ,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'delta_eff_stress'; tensor = True; component_index = 0; arr_name_plot = 'delta_eff_stress_XX,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'delta_eff_stress'; tensor = True; component_index = 1; arr_name_plot = 'delta_eff_stress_YY,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'delta_tot_stress'; tensor = True; component_index = 2; arr_name_plot = 'delta_tot_stress_ZZ,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'delta_tot_stress'; tensor = True; component_index = 0; arr_name_plot = 'delta_tot_stress_XX,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    arr_name = 'delta_tot_stress'; tensor = True; component_index = 1; arr_name_plot = 'delta_tot_stress_YY,MPa'; scale = 0.1
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index, scale))

    # --- geometry (from idata) ---
    rsv_top = idata.other.rsv_top
    rsv_bottom = idata.other.rsv_bottom
    rsv_xy = idata.other.rsv_xy
    prod_well = list(getattr(idata.other, 'prod_well_coords', []))  # [x, y, z1, z2]
    inj_well = list(getattr(idata.other, 'inj_well_coords', []))

    # X positions / colors of well marker lines (only the wells that exist)
    well_markers = []  # (x_well, color)
    if prod_well:
        well_markers.append((prod_well[0], 'red'))
    if inj_well:
        well_markers.append((inj_well[0], 'cyan'))
    # domain center = midpoint between wells (used for the central reference line and 1D 'center' profile)
    well_xs = [w[0] for w in (prod_well, inj_well) if w]
    center_x = float(np.mean(well_xs)) if well_xs else 0.0
    well_y = (prod_well[1] if prod_well else inj_well[1]) if (prod_well or inj_well) else 0.0
    # "right" 1D profile location = injection well (fallback to producer / center)
    right_well = inj_well or prod_well or [center_x, well_y]

    # reference points [x, y, label] for the black reference line / 1D profiles (from idata, optional)
    plot_points_xy = list(getattr(idata.other, 'points_xy', []))

    rsv_xy_plot_bnd = 5.0 * rsv_xy  # m. half-width of the plotted X-window (5000 for the default rsv_xy=1000)
    if use_mesh_bounds:  # use the actual mesh X-extent (+10% margin on each side) instead of the rsv_xy-based window
        x_margin = 0.2 * (block.bounds[1] - block.bounds[0])
        x_plot_min, x_plot_max = block.bounds[0] - x_margin, block.bounds[1] + x_margin
    else:
        x_plot_min, x_plot_max = -rsv_xy_plot_bnd, rsv_xy_plot_bnd

    print('mesh bounds:', [round(b, 1) for b in block.bounds])  # [xmin,xmax,ymin,ymax,zmin,zmax]
    print('plot bounds: X', round(x_plot_min, 1), round(x_plot_max, 1),
          'Z', round(block.bounds[4], 1), round(block.bounds[5], 1))

    for plot_config in plot_config_list:
        arr_name, tensor, arr_name_plot, contour, component_index, scale = plot_config

        if arr_name not in block.array_names: # skip temperature if not thermal model
            print('Warning: ', arr_name, 'not found in point data')
            continue

        print('Plotting from vtk: ', arr_name, 'component_index', component_index)
        block.set_active_scalars(None)
        if not tensor:
            block.set_active_scalars(arr_name, preference='point')
        else:
            block.set_active_tensors(arr_name, preference='point')

        # Create a slice
        slice_plane = block.slice(normal='y')
        #slice_plane = block  # no slice (plot in 3D)
        y_bnd = max(abs(block.bounds[2]), abs(block.bounds[3]))
        slice_plane = slice_plane.clip_box(
            bounds=[x_plot_min, x_plot_max, -y_bnd, y_bnd,
                    block.bounds[4], block.bounds[5]], invert=False)

        xmin_blk = x_plot_min
        xmax_blk = x_plot_max
        zmin_blk = block.bounds[4]
        zmax_blk = block.bounds[5]
        y_slice = block.center[1]
        out_png = os.path.join(output_dir_plots, arr_name_plot + "_slice.png")

        if use_mtri:  # optional matplotlib backend (centered axes + titles; see its drawbacks)
            plot_slice_matplotlib(slice_plane, arr_name, tensor, component_index, scale,
                                  arr_name_plot, contour, rsv_top, rsv_bottom,
                                  well_markers, plot_points_xy,
                                  xmin_blk, xmax_blk, zmin_blk, zmax_blk, out_png)
            continue

        # --- default pyvista rendering ---
        if tensor:
            slice_plane[arr_name_plot] = slice_plane[arr_name][:, component_index] * scale
        else:
            slice_plane[arr_name_plot] = slice_plane[arr_name] * scale

        plot_w, plot_h = 1024, 768  # fixed window so the camera aspect (hence centering) is known
        plotter = pv.Plotter(off_screen=True, window_size=(plot_w, plot_h)) # save without showing the GUI window

        # to plot with the same color if values are almost the same everywhere
        plot_rel_diff_threshold = 0.001
        values = np.array(slice_plane[arr_name_plot])
        if np.fabs(values.max() - values.min()) < plot_rel_diff_threshold:
            slice_plane[arr_name_plot][:] = values.min()

        plotter.add_mesh(slice_plane, scalars=arr_name_plot, show_edges=False,
                         scalar_bar_args={'vertical': True, 'position_x': 0.86, 'position_y': 0.25,
                                          'width': 0.05, 'height': 0.5, 'title': ''})

        if contour:
            slice_plane_points = slice_plane.cell_data_to_point_data()
            contours = slice_plane_points.contour(isosurfaces=20, scalars=arr_name_plot)
            if contours.n_cells > 0:  # skip empty plots
                plotter.add_mesh(contours, cmap="viridis", opacity=1, color="black")
                plotter.add_mesh(slice_plane_points.outline(), color="black")

        # horizontal reference lines at reservoir top and bottom
        for z_ref in [rsv_top, rsv_bottom]:
            plotter.add_mesh(pv.Line(pointa=(xmin_blk, y_slice, z_ref),
                                     pointb=(xmax_blk, y_slice, z_ref)),
                             color='white', line_width=0.5)
        # vertical lines at the wells (prod=red, inj=cyan) + black reference lines at idata.other.points_xy (if set)
        # well-line top: a bit above the visualized block top (10% of the visualized depth)
        z_well_top = zmin_blk - 0.1 * (zmax_blk - zmin_blk)
        well_lines = [(x_well, rsv_bottom, z_well_top, clr, 2) for x_well, clr in well_markers]
        for p in plot_points_xy:  # [x, y, label]
            well_lines.append((p[0], zmax_blk, zmin_blk, 'black', 1))
        for x_well, z2, z1, clr, lw in well_lines:
            plotter.add_mesh(pv.Line(pointa=(x_well, y_slice, z1),
                                     pointb=(x_well, y_slice, z2)),
                             color=clr, line_width=lw)
        # Centered orthographic XZ view. Margins around the [xmin_blk,xmax_blk] x [zmin_blk,zmax_blk]
        # window leave clear space for the axis labels (left/bottom) and the colorbar (right), so
        # they do not overlap the figure. (view_xz auto-camera renders this planar slice off-center.)
        cx, cz = 0.5 * (xmin_blk + xmax_blk), 0.5 * (zmin_blk + zmax_blk)
        aspect = plot_w / plot_h
        parallel_scale = max(0.5 * (zmax_blk - zmin_blk) * 1.35,   # vertical margin (X labels + titles)
                             0.5 * (xmax_blk - xmin_blk) / aspect * 1.6)  # horizontal margin (Z labels + colorbar)
        plotter.enable_parallel_projection()
        plotter.camera.focal_point = (cx, y_slice, cz)
        plotter.camera.position = (cx, y_slice + 1000.0, cz)  # look from +Y: world +X to the right
        plotter.camera.up = (0, 0, -1)                        # Z increases downward (depth)
        plotter.camera.parallel_scale = parallel_scale

        # world (x, z) -> viewport fraction (0..1). up=-Z, so larger depth is lower on screen.
        half_w = parallel_scale * aspect
        vp_x = lambda xw: 0.5 + (xw - cx) / (2.0 * half_w)
        vp_y = lambda zw: 0.5 + (cz - zw) / (2.0 * parallel_scale)
        # anchor ticks/labels to the visible data box edge = plot window clipped to the mesh
        # extent. (Using slice_plane.bounds is cell-extended beyond the clip; using the raw window
        # is outside the mesh when use_mesh_bounds adds margin - either leaves a gap to the figure.)
        data_x0 = max(xmin_blk, block.bounds[0])   # left edge
        data_z1 = min(zmax_blk, block.bounds[5])   # bottom (max-depth) edge
        data_left = vp_x(data_x0)
        data_bot = vp_y(data_z1)

        # round, nicely-spaced ticks (e.g. 0, 2000, 4000 instead of 1860, 3720)
        x_ticks = [t for t in MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(xmin_blk, xmax_blk)
                   if xmin_blk - 1 <= t <= xmax_blk + 1]
        z_ticks = [t for t in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(zmin_blk, zmax_blk)
                   if zmin_blk - 1 <= t <= zmax_blk + 1]

        # tick marks: short black lines on the data edges (X ticks below, Z ticks to the left)
        tick_vp = 0.02                           # tick length as a viewport fraction
        tick_world_z = tick_vp * 2.0 * parallel_scale
        tick_world_x = tick_vp * 2.0 * half_w
        for xt in x_ticks:
            plotter.add_mesh(pv.Line((xt, y_slice, data_z1), (xt, y_slice, data_z1 + tick_world_z)),
                             color='black', line_width=2.5)
        for zt in z_ticks:
            plotter.add_mesh(pv.Line((data_x0 - tick_world_x, y_slice, zt), (data_x0, y_slice, zt)),
                             color='black', line_width=2.5)

        # Axis labels as fixed viewport text (camera-aligned -> they match the data/ticks, and avoid
        # vtk show_bounds quirks). Z labels sit in the left margin, X labels in the bottom margin.
        z_num_x = max(data_left - 0.075, 0.04)
        for zt in z_ticks:
            plotter.add_text(f'{int(round(zt))}', position=(z_num_x, vp_y(zt) - 0.012),
                             font_size=12, viewport=True, color='black')
        # 'Z, m.' just left of the Z numbers (not stranded at the far window edge)
        plotter.add_text('Z, m.', position=(max(z_num_x - 0.07, 0.0), 0.5),
                         font_size=14, viewport=True, color='black')
        for xt in x_ticks:
            s = f'{int(round(xt))}'
            plotter.add_text(s, position=(vp_x(xt) - 0.009 * len(s), data_bot - 0.055),
                             font_size=12, viewport=True, color='black')
        plotter.add_text('X, m.', position=(0.46, data_bot - 0.115), font_size=14, viewport=True, color='black')
        plotter.add_text(arr_name_plot, position=(0.5 - 0.007 * len(arr_name_plot), 0.92),
                         font_size=14, viewport=True)  # caption / title
        plotter.show(screenshot=out_png)
        plotter.close()

        # Contour plot using matplotlib (resampling is done only when requested) ##########
        if plot_contours:
            # res_x, res_z are the resampling-grid resolution (number of interpolation
            # points along X and Z) - a rendering knob, not geometry, so it is not from idata.
            res_x, res_z = 1000, 1000
            xmin, xmax = slice_plane.points[:, 0].min(), slice_plane.points[:, 0].max()
            zmin, zmax = slice_plane.points[:, 2].min(), slice_plane.points[:, 2].max()
            y_mean = slice_plane.points[:, 1].mean()
            # Create a "Template" Grid (UniformGrid / ImageData) structured XY plane (at Y=0)
            grid = pv.ImageData(
                dimensions=(res_x, 1, res_z),
                spacing=((xmax - xmin)/(res_x-1), y_mean, (zmax - zmin)/(res_z-1)),
                origin=(xmin, y_mean, zmin)
            )

            # Sample the data from your original 'block' or 'slice'
            # interpolate values from the slice onto struct grid
            structured_resample = grid.sample(slice_plane)

            # reshape to 2D
            if tensor:
                values_2d = structured_resample[arr_name][:,component_index].reshape(res_x, res_z) * scale
            else:
                values_2d = structured_resample[arr_name].reshape(res_x, res_z) * scale

            # Get the X and Z coordinates as 2D arrays (matching the values)
            x_coords = structured_resample.points[:, 0].reshape(res_x, res_z)
            z_coords = structured_resample.points[:, 2].reshape(res_x, res_z)

            plt.figure(figsize=(8, 4))
            plt.contourf(x_coords, z_coords, values_2d, levels=30, cmap='viridis')
            plt.colorbar()
            for z_ref in [rsv_top, rsv_bottom]:
                plt.axhline(y=z_ref, color='white', linestyle='--', linewidth=1)
            for x_well, clr in well_markers:
                plt.axvline(x=x_well, color=clr, linestyle='--', linewidth=1)
            for p in plot_points_xy:  # black reference lines at idata.other.points_xy
                plt.axvline(x=p[0], color='black', linestyle='--', linewidth=1)
            plt.xlabel("X Axis")
            plt.ylabel("Depth, m.")
            plt.ylim(zmin_blk, zmax_blk)
            plt.gca().invert_yaxis()
            plot_suffix = "_contour.png"
            plt.savefig(os.path.join(output_dir_plots, arr_name_plot + plot_suffix))
            plt.close()

        # plot 1D #################################################################################
        if tstep_to_plot == -1:
            sample_resolution = 75  # number of point along Z for plotting
            # select a few evenly-spaced timestep indices (always include the 1-th and the last)
            n_t = len(reader.time_values)
            n_t_plot = 5
            t_indices_1d = sorted(set(
                [1] + list(np.linspace(0, n_t - 1, n_t_plot, dtype=int)) + [n_t - 1]))

            if 'stress' in arr_name:
                t_indices_1d = [tstep_to_plot]

            # Define line endpoints (x, y fixed; z varies) - from idata.other.points_xy when set
            if plot_points_xy:
                points_xy = plot_points_xy
            else:
                points_xy = [[center_x, well_y, 'center'],
                             [right_well[0], right_well[1], 'right']] # XY
            z1, z2 = zmin_blk, zmax_blk   # vertical extent (full mesh depth)
            for x0, y0, name in points_xy:
                p0 = (x0, y0, z1)
                p1 = (x0, y0, z2)
                plt.figure(figsize=(6, 6))
                for t_idx in t_indices_1d:
                    reader.set_active_time_value(reader.time_values[t_idx])
                    mesh_t = reader.read()
                    block_t = mesh_t[0] if isinstance(mesh_t, pv.MultiBlock) else mesh_t
                    sampled = block_t.sample_over_line(pointa=p0, pointb=p1, resolution=sample_resolution)
                    z = sampled.points[:, 2]
                    values = sampled.point_data[arr_name] * scale
                    t_days = reader.time_values[t_idx]
                    t_label = f't={t_days:.0f} d'
                    if len(values.shape) > 1:  # tensor: plot all components, label by component+time
                        for kk, comp in enumerate(['XX', 'YY', 'ZZ', 'YZ', 'XZ', 'XY']):
                            plt.plot(values[:, kk], z, "-o", markersize=2, label=f'{comp} {t_label}')
                    else:
                        plt.plot(values, z, "-o", markersize=2, label=t_label)
                arr_name_plot_1d = arr_name if len(values.shape) > 1 else arr_name_plot
                plt.xlabel(arr_name_plot_1d)
                plt.ylabel("Depth, m.")
                plt.ylim(zmin_blk, zmax_blk)
                plt.gca().invert_yaxis()
                plt.title(f"Vertical profile of {arr_name_plot_1d} at x={x0}, y={y0}")
                plt.grid(True)
                plt.minorticks_on()
                plt.grid(which='major', linestyle='-', linewidth=0.8)
                plt.grid(which='minor', linestyle=':', linewidth=0.5)
                plt.tight_layout()
                plt.legend(fontsize=7)
                plt.savefig(os.path.join(output_dir_plots, arr_name_plot_1d + '_vertic_line_' + name + '.png'))
                plt.close()
    print('Plotting from VTK is completed for', output_dir)
        ##################################################################################

if __name__ == "__main__":
    contour = False


    #model_folder = '17_17_15'
    model_folder = '41_41_66'
    #model_folder = '83_83_90'
    use_mesh_bounds = False  # True: plot the full mesh X-extent instead of the rsv_xy-based window

    #model_folder = 'case_1'
    #use_mesh_bounds = True #plot the full mesh X-extent instead of the rsv_xy-based window

    physics_type = 'single_phase_thermal'
    wells_type = 'doublet'
    output_dir = os.path.join('results', 'sol_cpp_' + physics_type + '_' + wells_type + '_' + model_folder)

    from set_case import set_input_data
    idata = set_input_data(case=model_folder, model_folder=model_folder,
                           physics_type=physics_type, wells_type=wells_type)

    timestep_list = [0, -1]
    #timestep_list = [4,40,80,120]

    for timestep in timestep_list:
        plot_vtk_pyvista(output_dir, idata, contour=contour, tstep_to_plot=timestep,
                         use_mesh_bounds=use_mesh_bounds)
