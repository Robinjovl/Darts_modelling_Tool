import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

# =========================================================
# basic helpers
# =========================================================

def nearest_time_index(time_vector, t_days):
    time_vector = np.asarray(time_vector, dtype=float)
    return int(np.argmin(np.abs(time_vector - t_days)))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_property_key(prop_dict, candidates):
    """
    Return first existing key from candidates.
    """
    for k in candidates:
        if k in prop_dict:
            return k
    raise KeyError(
        f"None of the candidate keys found: {candidates}\n"
        f"Available keys are: {list(prop_dict.keys())}"
    )


# =========================================================
# geometry helpers
# =========================================================

def get_cell_geometry(model, use_lgr=True):
    """
    Return x, y, z, dx, dy, dz arrays for all active cells.
    """
    res = model.reservoir

    x = np.asarray(res.cell_center_x, dtype=float)
    y = np.asarray(res.cell_center_y, dtype=float)
    z = np.asarray(res.cell_center_z, dtype=float)

    if use_lgr:
        dx = np.asarray(res.dx, dtype=float)
        dy = np.asarray(res.dy, dtype=float)
        dz = np.asarray(res.dz, dtype=float)
    else:
        dx = np.asarray(res.global_data["dx"], dtype=float).reshape(-1, order="F")
        dy = np.asarray(res.global_data["dy"], dtype=float).reshape(-1, order="F")
        dz = np.asarray(res.global_data["dz"], dtype=float).reshape(-1, order="F")

    return x, y, z, dx, dy, dz


def get_well_center_xy(model, well_name):
    """
    Average x,y coordinates of all perforations of one well.
    """
    res = model.reservoir
    x = np.asarray(res.cell_center_x, dtype=float)
    y = np.asarray(res.cell_center_y, dtype=float)

    well = None
    for w in res.wells:
        if w.name == well_name:
            well = w
            break
    if well is None:
        raise ValueError(f"Well '{well_name}' not found.")

    perf_cells = [p[1] for p in well.perforations]
    if len(perf_cells) == 0:
        raise ValueError(f"Well '{well_name}' has no perforations.")

    return float(np.mean(x[perf_cells])), float(np.mean(y[perf_cells]))


def get_inj_prod_line(model, inj_name="I1", prod_name="P1"):
    """
    Return x_prod, x_inj, y_line.
    """
    x_inj, y_inj = get_well_center_xy(model, inj_name)
    x_prod, y_prod = get_well_center_xy(model, prod_name)
    y_line = 0.5 * (y_inj + y_prod)
    return x_prod, x_inj, y_line


# =========================================================
# 2D XY map plotting
# =========================================================

def extract_xy_slice_cells(model, values, depth=2035.0, use_lgr=True):
    """
    Extract one XY layer by selecting cells closest to given depth.
    """
    x, y, z, dx, dy, dz = get_cell_geometry(model, use_lgr=use_lgr)
    v = np.asarray(values, dtype=float)

    # choose cells whose center z is closest to requested depth
    unique_z = np.unique(np.round(z, 8))
    z_sel = unique_z[np.argmin(np.abs(unique_z - depth))]
    mask = np.isclose(z, z_sel, atol=1e-8)

    return x[mask], y[mask], dx[mask], dy[mask], v[mask], z_sel


def plot_xy_patch_map(
    model,
    values,
    depth=2035.0,
    use_lgr=True,
    savepath="xy_map.png",
    title="",
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    colorbar_label="",
):
    """
    Plot cell-based XY map using rectangles.
    """
    x, y, dx, dy, v, z_sel = extract_xy_slice_cells(
        model, values, depth=depth, use_lgr=use_lgr
    )

    if len(x) == 0:
        raise ValueError(f"No cells found at requested depth ~ {depth} m.")

    if vmin is None:
        vmin = float(np.min(v))
    if vmax is None:
        vmax = float(np.max(v))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)

    patches = []
    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        patches.append(Rectangle((xi - 0.5 * dxi, yi - 0.5 * dyi), dxi, dyi))

    pc = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        linewidth=0.0,
    )
    pc.set_array(v)
    ax.add_collection(pc)

    ax.set_xlim(np.min(x - 0.5 * dx), np.max(x + 0.5 * dx))
    ax.set_ylim(np.min(y - 0.5 * dy), np.max(y + 0.5 * dy))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title if title else f"XY map at z ~ {z_sel:.2f} m")

    cbar = fig.colorbar(pc, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_difference_map(
    model,
    time_vector,
    prop_array,
    year1,
    year2,
    depth=2035.0,
    use_lgr=True,
    savepath="diff_map.png",
    title="",
    cmap="coolwarm",
    symmetric=True,
    colorbar_label="",
):
    """
    Plot difference map: prop(year2) - prop(year1)
    """
    i1 = nearest_time_index(time_vector, year1 * 365.0)
    i2 = nearest_time_index(time_vector, year2 * 365.0)

    diff = np.asarray(prop_array[i2], dtype=float) - np.asarray(prop_array[i1], dtype=float)

    if symmetric:
        vmax = float(np.max(np.abs(diff)))
        vmin = -vmax
    else:
        vmin = float(np.min(diff))
        vmax = float(np.max(diff))

    plot_xy_patch_map(
        model=model,
        values=diff,
        depth=depth,
        use_lgr=use_lgr,
        savepath=savepath,
        title=title if title else f"Difference map: year {year2} - year {year1}",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
    )


# =========================================================
# 1D line extraction
# =========================================================

def extract_1d_line_between_wells(
    model,
    values,
    depth=2035.0,
    y0=None,
    x_from=None,
    x_to=None,
    use_lgr=True,
):
    """
    Extract 1D property profile along a horizontal line between injector and producer
    in the layer closest to `depth`.

    For LGR:
    - coarse cells: pick nearest coarse y-column
    - fine cells:   pick nearest fine y-column
    and merge them.
    """
    x, y, z, dx, dy, dz = get_cell_geometry(model, use_lgr=use_lgr)
    v = np.asarray(values, dtype=float)

    # choose closest layer
    unique_z = np.unique(np.round(z, 8))
    z_sel = unique_z[np.argmin(np.abs(unique_z - depth))]
    depth_mask = np.isclose(z, z_sel, atol=1e-8)

    if y0 is None:
        _, _, y0 = get_inj_prod_line(model)

    dy_max = float(np.max(dy))
    is_coarse = np.isclose(dy, dy_max, atol=1e-10)
    is_fine = ~is_coarse

    mask = np.zeros_like(y, dtype=bool)

    # coarse strip
    if np.any(is_coarse & depth_mask):
        y_coarse = np.unique(np.round(y[is_coarse & depth_mask], 10))
        y_coarse_sel = y_coarse[np.argmin(np.abs(y_coarse - y0))]
        mask |= (is_coarse & depth_mask & np.isclose(y, y_coarse_sel, atol=1e-8))

    # fine strip
    if np.any(is_fine & depth_mask):
        y_fine = np.unique(np.round(y[is_fine & depth_mask], 10))
        y_fine_sel = y_fine[np.argmin(np.abs(y_fine - y0))]
        mask |= (is_fine & depth_mask & np.isclose(y, y_fine_sel, atol=1e-8))

    if x_from is not None:
        mask &= (x >= min(x_from, x_to))
    if x_to is not None:
        mask &= (x <= max(x_from, x_to))

    x_line = x[mask]
    y_line = y[mask]
    v_line = v[mask]

    if len(x_line) == 0:
        raise ValueError("No cells found for 1D line extraction.")

    # sort by x
    idx = np.argsort(x_line)
    x_line = x_line[idx]
    y_line = y_line[idx]
    v_line = v_line[idx]

    # deduplicate nearly identical x if needed (keep mean)
    x_round = np.round(x_line, 6)
    uniq_x = np.unique(x_round)

    x_out = []
    v_out = []
    for xu in uniq_x:
        m = np.isclose(x_round, xu)
        x_out.append(np.mean(x_line[m]))
        v_out.append(np.mean(v_line[m]))

    return np.asarray(x_out), np.asarray(v_out), float(z_sel)


def plot_1d_profiles_between_wells(
    model,
    time_vector,
    prop_array,
    years=(1, 5, 10, 15, 20, 30, 40, 50),
    depth=2035.0,
    use_lgr=True,
    savepath="profile_1d.png",
    title="",
    ylabel="",
    inj_name="I1",
    prod_name="P1",
):
    """
    Plot 1D profiles between producer and injector for several timesteps.
    """
    x_prod, x_inj, y_line = get_inj_prod_line(model, inj_name=inj_name, prod_name=prod_name)
    x_min = min(x_prod, x_inj)
    x_max = max(x_prod, x_inj)

    plt.figure(figsize=(8, 5), dpi=160)

    for yr in years:
        idx = nearest_time_index(time_vector, yr * 365.0)
        values = np.asarray(prop_array[idx], dtype=float)

        x_line, v_line, z_sel = extract_1d_line_between_wells(
            model=model,
            values=values,
            depth=depth,
            y0=y_line,
            x_from=x_min,
            x_to=x_max,
            use_lgr=use_lgr,
        )

        dist = x_line - x_min
        plt.plot(dist, v_line, label=f"{yr} yr")

    plt.xlabel("Distance from producer to injector [m]")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    else:
        plt.title(f"1D profile between wells at z ~ {z_sel:.2f} m")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


# =========================================================
# high-level batch wrappers
# =========================================================

def make_difference_maps_batch(
    model,
    time_vector,
    prop_array,
    out_dir,
    prop_name,
    year_pairs,
    depth=2035.0,
    use_lgr=True,
    cmap="coolwarm",
    symmetric=True,
    colorbar_label="",
):
    ensure_dir(out_dir)

    for y1, y2 in year_pairs:
        savepath = os.path.join(out_dir, f"{prop_name}_diff_y{y2}-y{y1}.png")
        title = f"{prop_name}: year {y2} - year {y1}"
        plot_difference_map(
            model=model,
            time_vector=time_vector,
            prop_array=prop_array,
            year1=y1,
            year2=y2,
            depth=depth,
            use_lgr=use_lgr,
            savepath=savepath,
            title=title,
            cmap=cmap,
            symmetric=symmetric,
            colorbar_label=colorbar_label,
        )


def make_profiles_batch(
    model,
    time_vector,
    prop_array,
    out_dir,
    prop_name,
    years,
    depth=2035.0,
    use_lgr=True,
    ylabel="",
):
    ensure_dir(out_dir)

    savepath = os.path.join(out_dir, f"{prop_name}_1d_profiles_between_wells.png")
    title = f"{prop_name} profiles between injector and producer"
    plot_1d_profiles_between_wells(
        model=model,
        time_vector=time_vector,
        prop_array=prop_array,
        years=years,
        depth=depth,
        use_lgr=use_lgr,
        savepath=savepath,
        title=title,
        ylabel=ylabel,
    )
