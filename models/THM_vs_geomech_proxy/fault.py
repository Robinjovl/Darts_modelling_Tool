"""
Fault stability post-processing for the THM-vs-geomech-proxy model.

Workflow
--------
1. During the run, `reservoir.save_fault_traction` (called from write_to_vtk)
   writes for every report step `results/<case>/fault<step>.vtu`: the 2D fault
   surface taken from the `*_fault.msh` companion mesh (faces tagged
   FAULT = 9991, see gen_fault_msh_no_damage_zone.py) with the cell data
     traction  (3)  total-stress traction on the fault [bar], compression positive
     normal    (3)  unit fault normal the traction refers to
     pressure       pore pressure at the fault, mean of both sides [bar]
     temperature    temperature at the fault, mean of both sides [K] (thermal runs only)
   and the time series `results/<case>/fault.pvd`.

2. This script reads those files, evaluates the Mohr-Coulomb criterion and the
   fault stability potential and appends them to the same fault<step>.vtu:
     sigma_n        normal stress, compression positive [bar]
     sigma_n_eff    effective normal stress sigma_n - p [bar]
     tau            shear stress magnitude [bar]
     mcc            tau - (cohesion + friction * sigma_n_eff) [bar], > 0 means slip
     FSP            tau / sigma_n_eff (NaN where sigma_n_eff <= 0, i.e. opened fault)
     slip           1 where mcc > 0, else 0
   It also writes fault_slip.csv and fault_slip_vs_time.png into the case folder.

Usage (from the model folder):

    python fault.py [case_dir] [--friction 0.6] [--cohesion 0.0]
"""

import argparse
import glob
import os
import re

import meshio
import numpy as np

FRAC_TAG = 9991  # physical tag assigned to the fault surfaces in *_fault.msh
FAULT_CELL_TYPES = ('triangle', 'quad')


# ---------------------------------------------------------------------------
# Fault geometry from the *_fault.msh companion mesh
# ---------------------------------------------------------------------------
def polygon_areas_normals(faces_xyz):
    """Areas and unit normals of planar polygons (n, k, 3) via fan triangulation."""
    fan = faces_xyz - faces_xyz[:, 0:1, :]
    cross = np.cross(fan[:, 1:-1, :], fan[:, 2:, :]).sum(axis=1)
    norm = np.linalg.norm(cross, axis=1)
    return 0.5 * norm, cross / norm[:, None]


def read_fault_mesh(mesh_filename, frac_tag=FRAC_TAG):
    """
    Read the fault surface from a `*_fault.msh` mesh (`mesh.msh` is accepted,
    the `_fault` suffix is added if missing).

    Returns a dict with
      points  (m, 3)  nodes used by the fault faces
      cells   list of (cell type, connectivity) blocks with the fault faces only
      centers (n, 3), areas (n,), normals (n, 3)  per face, in block order.
    Face normals are oriented to one side of the fault: along the normal of the
    best-fit plane through the face centers, taken with a non-negative z (depth)
    component.
    """
    base, ext = os.path.splitext(mesh_filename)
    if not base.endswith('_fault'):
        mesh_filename = base + '_fault' + ext
    msh = meshio.read(mesh_filename)

    blocks = []
    for block, tags in zip(msh.cells, msh.cell_data['gmsh:physical']):
        if block.type in FAULT_CELL_TYPES:
            faces = block.data[np.asarray(tags) == frac_tag]
            if len(faces):
                blocks.append((block.type, faces))
    assert blocks, 'no fault faces with tag %d found in %s' % (frac_tag, mesh_filename)

    # keep only the nodes of the fault faces
    used = np.unique(np.concatenate([faces.ravel() for _, faces in blocks]))
    new_id = np.full(len(msh.points), -1, dtype=np.int64)
    new_id[used] = np.arange(len(used))
    points = msh.points[used]
    cells = [(cell_type, new_id[faces]) for cell_type, faces in blocks]

    centers = np.concatenate([points[faces].mean(axis=1) for _, faces in cells])
    areas, normals = map(np.concatenate, zip(*[polygon_areas_normals(points[faces]) for _, faces in cells]))

    ref = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)[2][-1]
    if ref[2] < 0:
        ref = -ref
    normals *= np.where(normals @ ref < 0, -1.0, 1.0)[:, None]
    return {'points': points, 'cells': cells, 'centers': centers, 'areas': areas, 'normals': normals}


def split_by_blocks(values, cells):
    """Split a per-face array into the per-block list expected by meshio cell_data."""
    return np.split(values, np.cumsum([len(faces) for _, faces in cells])[:-1])


# ---------------------------------------------------------------------------
# Mohr-Coulomb criterion and FSP
# ---------------------------------------------------------------------------
def fault_stability(traction, normal, pressure, friction=0.6, cohesion=0.):
    """
    Mohr-Coulomb slip criterion and fault stability potential per fault face:

        sigma_n     = t . n                       normal stress (compression positive)
        tau         = |t - sigma_n n|             shear stress magnitude
        sigma_n_eff = sigma_n - p                 effective normal stress
        mcc         = tau - (cohesion + friction * sigma_n_eff)     > 0: slip
        FSP         = tau / sigma_n_eff           NaN where sigma_n_eff <= 0 (opened fault)

    t is the total-stress traction on the fault, n the unit fault normal, p the pore
    pressure. The effective normal stress uses the full pore pressure (Terzaghi): the
    fault has its own constitutive behavior, so the Biot coefficient of the matrix does
    not apply.

    :param traction: (n, 3) total-stress traction on the fault, compression positive
    :param normal: (n, 3) unit fault normals
    :param pressure: (n,) pore pressure, same units as traction
    :param friction: friction coefficient
    :param cohesion: cohesion, same units as traction
    :return: dict of (n,) arrays sigma_n, sigma_n_eff, tau, mcc, FSP, slip
    """
    sigma_n = np.einsum('ij,ij->i', traction, normal)
    tau = np.linalg.norm(traction - sigma_n[:, None] * normal, axis=1)
    sigma_n_eff = sigma_n - pressure
    mcc = tau - (cohesion + friction * sigma_n_eff)
    with np.errstate(divide='ignore', invalid='ignore'):
        fsp = np.where(sigma_n_eff > 0, tau / sigma_n_eff, np.nan)
    return {'sigma_n': sigma_n, 'sigma_n_eff': sigma_n_eff, 'tau': tau,
            'mcc': mcc, 'FSP': fsp, 'slip': (mcc > 0).astype(np.int32)}


# ---------------------------------------------------------------------------
# Post-processing of a finished run
# ---------------------------------------------------------------------------
def _step_of(path):
    match = re.search(r'fault(\d+)\.vtu$', os.path.basename(path))
    return int(match.group(1)) if match else -1


def _fault_files(case_dir):
    """fault<step>.vtu files of a run, sorted by step."""
    return sorted((f for f in glob.glob(os.path.join(case_dir, 'fault*.vtu')) if _step_of(f) >= 0), key=_step_of)


def _read_pvd_times(pvd_filename):
    """Map step -> simulation time [days] from a .pvd file (empty if absent)."""
    times = {}
    if os.path.exists(pvd_filename):
        with open(pvd_filename) as f:
            for match in re.finditer(r'timestep="([^"]+)"\s+file="[^"]*?(\d+)\.vtu"', f.read()):
                times[int(match.group(2))] = float(match.group(1))
    return times


SERIES_COLUMNS = ['step', 'time_days', 'slip_area_m2', 'slip_area_fraction',
                  'fsp_mean', 'fsp_max', 'delta_fsp_max', 'mcc_max_bar']


def print_face_computation(traction, normal, pressure, friction=0.6, cohesion=0., label=''):
    """Print the FSP computation for one fault face step by step (units: bar)."""
    t, n, p = np.asarray(traction, dtype=float), np.asarray(normal, dtype=float), float(pressure)
    sigma_n = t @ n
    tau_vec = t - sigma_n * n
    tau = np.linalg.norm(tau_vec)
    sigma_n_eff = sigma_n - p
    fsp =tau / sigma_n_eff if sigma_n_eff > 0 else np.nan
    mcc = tau - (cohesion + friction * sigma_n_eff)
    print('FSP computation %s' % label)
    print('  traction t (compression +)   = [%.3f, %.3f, %.3f] bar' % tuple(t))
    print('  normal n                     = [%.4f, %.4f, %.4f]' % tuple(n))
    print('  sigma_n = t . n              = %.3f bar' % sigma_n)
    print('  tau_vec = t - sigma_n n      = [%.3f, %.3f, %.3f] bar' % tuple(tau_vec))
    print('  tau = |tau_vec|              = %.3f bar' % tau)
    print('  p                            = %.3f bar' % p)
    print('  sigma_n_eff = sigma_n - p    = %.3f - %.3f = %.3f bar' % (sigma_n, p, sigma_n_eff))
    print('  FSP = tau / sigma_n_eff      = %.3f / %.3f = %.4f' % (tau, sigma_n_eff, fsp))
    print('  mcc = tau - (c + mu sigma_n_eff) = %.3f - (%g + %g * %.3f) = %.3f bar (%s)'
          % (tau, cohesion, friction, sigma_n_eff, mcc, 'slip' if mcc > 0 else 'stable'))


def fault_series(case_dir, friction=0.6, cohesion=0., append=True):
    """
    Evaluate `fault_stability` for every fault<step>.vtu of a run.

    :param append: write the results and delta_FSP = FSP - FSP(first step) into the files
                   as cell data (arrays of the same name are overwritten); with False the
                   files are only read, e.g. to compare several friction values
    :return: dict of per-step arrays named as in SERIES_COLUMNS (fsp_mean is
             area weighted) plus 'fault_area_m2'
    """
    files = _fault_files(case_dir)
    if not files:
        raise FileNotFoundError('no fault<step>.vtu found in %s' % case_dir)
    times = _read_pvd_times(os.path.join(case_dir, 'fault.pvd'))

    rows, fsp0 = [], None
    for f in files:
        m = meshio.read(f)
        data = {name: np.concatenate(arrays) for name, arrays in m.cell_data.items()}
        res = fault_stability(data['traction'], data['normal'], data['pressure'],
                              friction=friction, cohesion=cohesion)
        fsp0 = res['FSP'] if fsp0 is None else fsp0
        res['delta_FSP'] = res['FSP'] - fsp0
        cells = [(block.type, block.data) for block in m.cells]
        if append:
            for name, values in res.items():
                m.cell_data[name] = split_by_blocks(values, cells)
            meshio.write(f, m)

        areas = np.concatenate([polygon_areas_normals(m.points[faces])[0] for _, faces in cells])
        slip_area = areas[res['slip'] > 0].sum()
        ok = np.isfinite(res['FSP'])
        step = _step_of(f)
        rows.append([step, times.get(step, np.nan), slip_area, slip_area / areas.sum(),
                     (res['FSP'][ok] * areas[ok]).sum() / areas[ok].sum() if ok.any() else np.nan,
                     res['FSP'][ok].max() if ok.any() else np.nan,
                     np.nanmax(res['delta_FSP']) if np.isfinite(res['delta_FSP']).any() else np.nan,
                     res['mcc'].max()])

    rows = np.array(rows, dtype=float)
    series = {name: rows[:, i] for i, name in enumerate(SERIES_COLUMNS)}
    series['fault_area_m2'] = areas.sum()
    return series


def _time_axis(series):
    if np.isfinite(series['time_days']).all():
        return series['time_days'] / 365.25, 'time [years]'
    return series['step'], 'report step'


def postprocess_case(case_dir, friction=0.6, cohesion=0., verbose=True):
    """
    Append FSP, delta_FSP and the Mohr-Coulomb criterion to every fault<step>.vtu
    in `case_dir` and write the slip time series: fault_slip.csv and
    fault_slip_vs_time.png (FSP and slipping area vs time). With verbose=True the
    computation is also printed step by step for the face with the largest FSP of
    the last step.
    """
    series = fault_series(case_dir, friction=friction, cohesion=cohesion, append=True)
    if verbose:
        m = meshio.read(_fault_files(case_dir)[-1])
        data = {name: np.concatenate(arrays) for name, arrays in m.cell_data.items()}
        i = np.nanargmax(data['FSP']) if np.isfinite(data['FSP']).any() else 0
        print_face_computation(data['traction'][i], data['normal'][i], data['pressure'][i], friction=friction,
                               cohesion=cohesion, label='(face %d of step %d)'
                               % (i, int(series['step'][-1])))
        print('%s: friction=%g, cohesion=%g bar, fault area %.3e m2'
              % (case_dir, friction, cohesion, series['fault_area_m2']))
        for i in range(series['step'].size):
            print('  step %4d: FSP mean/max %.4f/%.4f  dFSP max %.4f  mcc max %9.3f bar  slip area %.3e m2 (%.1f%%)'
                  % tuple([series[c][i] for c in ['step', 'fsp_mean', 'fsp_max', 'delta_fsp_max', 'mcc_max_bar',
                                                   'slip_area_m2']] + [100 * series['slip_area_fraction'][i]]))

    csv_filename = os.path.join(case_dir, 'fault_slip.csv')
    np.savetxt(csv_filename, np.column_stack([series[c] for c in SERIES_COLUMNS]), delimiter=',',
               header=','.join(SERIES_COLUMNS), comments='',
               fmt=['%d', '%.6f', '%.6e', '%.6f', '%.6f', '%.6f', '%.6f', '%.6e'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x, xlabel = _time_axis(series)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax1.plot(x, series['fsp_mean'], '-o', label='area-weighted mean')
    ax1.plot(x, series['fsp_max'], '-s', label='max')
    ax1.axhline(friction, color='k', ls='--', lw=1, label='friction')
    ax1.set_ylabel('FSP [-]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(x, series['slip_area_m2'] * 1e-6, '-o', color='C3')
    ax2.set_ylabel('slipping area [km$^2$]')
    ax2.set_xlabel(xlabel)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    percent = ax2.secondary_yaxis('right', functions=(lambda a: 1e8 * a / series['fault_area_m2'],
                                                      lambda q: 1e-8 * q * series['fault_area_m2']))
    percent.set_ylabel('[% of fault area]')
    ax1.set_title('friction %g, cohesion %g bar' % (friction, cohesion))
    fig.tight_layout()
    fig.savefig(os.path.join(case_dir, 'fault_slip_vs_time.png'), dpi=150)
    plt.close(fig)
    if verbose:
        print('slip time series ->', csv_filename, 'and fault_slip_vs_time.png')
    return series


def compare_cases(case_dirs, labels=None, frictions=(0.6,), cohesion=0.,
                  png_filename='fault_slip_comparison.png', verbose=True):
    """
    Compare the slipping area and FSP (max, area-weighted mean) vs time of several
    runs, for one or more friction values (the fault files are only read).
    Saves png_filename and a csv with the same basename.
    :return: {(label, friction): series}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = labels or [os.path.basename(os.path.normpath(d)) for d in case_dirs]
    results = {}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for i, (case_dir, label) in enumerate(zip(case_dirs, labels)):
        for j, friction in enumerate(frictions):
            series = fault_series(case_dir, friction=friction, cohesion=cohesion, append=False)
            results[(label, friction)] = series
            x, xlabel = _time_axis(series)
            style = dict(color='C%d' % i, ls=['-', '--', ':', '-.'][j % 4])
            axes[0].plot(x, series['slip_area_m2'] * 1e-6, label='%s, friction %g' % (label, friction), **style)
        axes[1].plot(x, series['fsp_max'], color='C%d' % i, label=label)
        axes[2].plot(x, series['fsp_mean'], color='C%d' % i, label=label)
    fig.suptitle('cohesion %g bar' % cohesion)
    axes[0].set_ylabel('slipping area [km$^2$]')
    axes[1].set_ylabel('FSP max [-]')
    axes[2].set_ylabel('FSP area-weighted mean [-]')
    axes[2].set_xlabel(xlabel)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_filename, dpi=150)
    plt.close(fig)

    header = 'case,friction,' + ','.join(SERIES_COLUMNS)
    with open(os.path.splitext(png_filename)[0] + '.csv', 'w') as f:
        f.write(header + '\n')
        for (label, friction), series in results.items():
            for k in range(series['step'].size):
                f.write('%s,%g,' % (label, friction) + ','.join('%.8g' % series[c][k] for c in SERIES_COLUMNS) + '\n')
    if verbose:
        print('%-40s %8s %14s %14s %10s %10s' % ('case', 'friction', 'slip area0 km2', 'slip areaN km2',
                                                 'FSP max0', 'FSP maxN'))
        for (label, friction), s in results.items():
            print('%-40s %8g %14.4f %14.4f %10.4f %10.4f' % (label, friction, s['slip_area_m2'][0] * 1e-6,
                                                             s['slip_area_m2'][-1] * 1e-6, s['fsp_max'][0],
                                                             s['fsp_max'][-1]))
        print('comparison ->', png_filename)
    return results


# ---------------------------------------------------------------------------
# 2D maps of fault fields
# ---------------------------------------------------------------------------
FIELD_UNITS = {'FSP': '-', 'delta_FSP': '-', 'slip': '-', 'mcc': 'bar', 'sigma_n': 'bar', 'sigma_n_eff': 'bar',
               'tau': 'bar', 'pressure': 'bar'}
SIGNED_FIELDS = ['delta_FSP']  # drawn with a diverging colormap symmetric around 0


def fault_plane_axes(normal):
    """Unit along-strike and down-dip vectors of a plane with the given normal (z is depth)."""
    n = normal / np.linalg.norm(normal)
    strike = np.cross([0., 0., 1.], n)
    if np.linalg.norm(strike) < 1e-8:  # horizontal plane
        strike = np.array([1., 0., 0.])
    strike /= np.linalg.norm(strike)
    dip = np.cross(n, strike)
    return strike, (dip if dip[2] >= 0 else -dip)


def plot_fault_field(vtu_filename, field='FSP', png_filename=None, vmin=None, vmax=None,
                     cmap='viridis', title=None):
    """
    Save a 2D map of a per-face field of fault<step>.vtu, drawn on the fault faces
    unfolded onto the fault plane (along strike vs down dip). Faces with NaN
    (e.g. FSP of opened faces) are grey.

    :param png_filename: output file, default <vtu without extension>_<field>.png
    :return: png_filename
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    m = meshio.read(vtu_filename)
    if field not in m.cell_data:
        raise KeyError('%s is not in %s (run postprocess_case first?)' % (field, vtu_filename))
    values = np.concatenate(m.cell_data[field]).astype(float)
    strike, dip = fault_plane_axes(np.concatenate(m.cell_data['normal']).mean(axis=0))
    uv = np.column_stack([m.points @ strike, m.points @ dip])
    uv -= uv.min(axis=0)
    polygons = [uv[face] for block in m.cells for face in block.data]

    finite = values[np.isfinite(values)]
    vmin = (finite.min() if finite.size else 0.) if vmin is None else vmin
    vmax = (finite.max() if finite.size else 1.) if vmax is None else vmax
    color_map = matplotlib.colormaps[cmap].copy()
    color_map.set_bad('lightgrey')

    width, height = uv.max(axis=0)
    fig, ax = plt.subplots(figsize=(10, min(8., max(3., 10 * height / width + 1.5))))
    faces = PolyCollection(polygons, array=np.ma.masked_invalid(values), cmap=color_map, edgecolors='face')
    faces.set_clim(vmin, vmax)
    ax.add_collection(faces)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # deeper part of the fault at the bottom
    ax.set_aspect('equal')
    ax.set_xlabel('along strike [m]')
    ax.set_ylabel('down dip [m]')
    ax.set_title(title if title is not None else '%s  %s' % (field, os.path.basename(vtu_filename)))
    fig.colorbar(faces, ax=ax, label='%s [%s]' % (field, FIELD_UNITS.get(field, '-')), shrink=0.9)
    fig.tight_layout()

    png_filename = png_filename or os.path.splitext(vtu_filename)[0] + '_%s.png' % field
    fig.savefig(png_filename, dpi=150)
    plt.close(fig)
    return png_filename


def plot_fault_case(case_dir, fields=('FSP',), same_scale=True, out_dir=None, verbose=True):
    """
    Save 2D maps of `fields` for every fault<step>.vtu of a run into
    <case_dir>/fault_plots/<field>_step<NNNN>.png. With same_scale=True all steps
    of a field share one color range, so the maps are comparable in time.
    """
    files = _fault_files(case_dir)
    if not files:
        print('[WARN] no fault<step>.vtu found in', case_dir)
        return []
    times = _read_pvd_times(os.path.join(case_dir, 'fault.pvd'))
    out_dir = out_dir or os.path.join(case_dir, 'fault_plots')
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for field in fields:
        vmin = vmax = None
        if same_scale:
            values = np.concatenate([np.concatenate(meshio.read(f).cell_data[field]) for f in files]).astype(float)
            values = values[np.isfinite(values)]
            if values.size:
                vmin, vmax = values.min(), values.max()
                if field in SIGNED_FIELDS:
                    vmax = max(abs(vmin), abs(vmax), 1e-12)
                    vmin = -vmax
        cmap = 'coolwarm' if field in SIGNED_FIELDS else 'viridis'
        for f in files:
            step = _step_of(f)
            title = '%s, step %d' % (field, step) + (', t = %g days' % times[step] if step in times else '')
            saved.append(plot_fault_field(f, field, os.path.join(out_dir, '%s_step%04d.png' % (field, step)),
                                          vmin=vmin, vmax=vmax, cmap=cmap, title=title))
    if verbose:
        print('%d fault maps -> %s' % (len(saved), out_dir))
    return saved


# ---------------------------------------------------------------------------
# 1D profiles along the fault dip
# ---------------------------------------------------------------------------
DIP_PROFILE_FIELDS = ['pressure', 'temperature', 'sigma_n', 'sigma_n_eff', 'tau', 'mcc', 'FSP']


def plot_fault_dip_profiles(case_dir, strike=None, steps=None, friction=0.6, cohesion=0.,
                            png_filename=None, verbose=True, point=None):
    """
    Plot pore pressure, temperature (if saved, i.e. thermal runs), normal stress (total
    and effective), shear stress, Coulomb stress mcc = tau - (cohesion + friction *
    sigma_n_eff) and FSP vs depth along the fault dip, for the column of fault faces at
    one along-strike position.
    The quantities are recomputed from traction/normal/pressure with the given
    friction and cohesion. A csv with the same basename holds the profiles.

    :param strike: along-strike position [m], same coordinate as the 2D maps (0 at the
                   fault edge); default: position of the face with the largest FSP at
                   the last step
    :param point: [x, y, z] (e.g. a well perforation) whose along-strike position is
                  used instead of `strike`
    :param steps: report steps to draw; default: first, middle and last
    :param png_filename: default <case_dir>/fault_plots/dip_profile_strike<strike>m.png
    :return: png_filename
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    files = {_step_of(f): f for f in _fault_files(case_dir)}
    if not files:
        raise FileNotFoundError('no fault<step>.vtu found in %s' % case_dir)
    all_steps = sorted(files)
    steps = sorted(set(steps)) if steps else sorted({all_steps[0], all_steps[len(all_steps) // 2], all_steps[-1]})
    unknown = set(steps) - set(files)
    if unknown:
        raise ValueError('steps %s are not in %s' % (sorted(unknown), case_dir))
    times = _read_pvd_times(os.path.join(case_dir, 'fault.pvd'))

    def evaluate(step):
        m = meshio.read(files[step])
        data = {name: np.concatenate(arrays) for name, arrays in m.cell_data.items()}
        res = fault_stability(data['traction'], data['normal'], data['pressure'],
                              friction=friction, cohesion=cohesion)
        res['pressure'] = data['pressure']
        if 'temperature' in data:
            res['temperature'] = data['temperature']
        return m, data, res

    # geometry (fixed in time): along-strike extent and center of every face
    m, data, res_last = evaluate(all_steps[-1])
    strike_dir, _ = fault_plane_axes(data['normal'].mean(axis=0))
    u_nodes = m.points @ strike_dir
    u_origin = u_nodes.min()
    u_nodes -= u_origin
    faces = [face for block in m.cells for face in block.data]
    u_min = np.array([u_nodes[face].min() for face in faces])
    u_max = np.array([u_nodes[face].max() for face in faces])
    depth = np.array([m.points[face, 2].mean() for face in faces])

    if point is not None:
        strike = float(np.asarray(point, dtype=float) @ strike_dir - u_origin)
    if strike is None:
        ok = np.flatnonzero(np.isfinite(res_last['FSP']))
        i = ok[np.argmax(res_last['FSP'][ok])] if ok.size else 0
        strike = 0.5 * (u_min[i] + u_max[i])
    column = np.flatnonzero((u_min <= strike) & (strike < u_max))
    if column.size == 0:  # at the far edge of the fault
        column = np.flatnonzero(np.isclose(u_max, u_max.max()) if strike >= u_max.max() else
                                np.isclose(u_min, u_min.min()))
    column = column[np.argsort(depth[column])]
    z = depth[column]

    panels = [(['pressure'], 'pore pressure [bar]'),
              (['temperature'], 'temperature [K]'),
              (['sigma_n', 'sigma_n_eff'],
               "normal stress [bar]\nsolid: $\\sigma_n$, dashed: $\\sigma'_n$"),
              (['tau'], 'shear stress $\\tau$ [bar]'),
              (['mcc'], 'Coulomb stress [bar]'),
              (['FSP'], 'FSP [-]')]
    panels = [panel for panel in panels if panel[0][0] in res_last]  # no temperature in isothermal runs
    fields = [name for name in DIP_PROFILE_FIELDS if name in res_last]
    panel_of = {names[0]: k for k, (names, _) in enumerate(panels)}
    fig, axes = plt.subplots(1, len(panels), sharey=True, figsize=(3.4 * len(panels), 6.5))
    colors = matplotlib.colormaps['viridis'](np.linspace(0., 0.9, len(steps)))
    table = []
    for step, color in zip(steps, colors):
        res = res_last if step == all_steps[-1] else evaluate(step)[2]
        label = 'step %d' % step + (', t = %.2f y' % (times[step] / 365.25) if step in times else '')
        for ax, (names, _) in zip(axes, panels):
            for name, ls in zip(names, ['-', '--']):
                ax.plot(res[name][column], z, ls, marker='o' if ls == '-' else None, ms=3, color=color,
                        label=label if ls == '-' else None)
        table.append(np.column_stack([np.full(z.size, step), z] + [res[name][column] for name in fields]))

    axes[panel_of['mcc']].axvline(0., color='k', lw=1, ls=':')
    axes[panel_of['FSP']].axvline(friction, color='k', lw=1, ls='--', label='friction %g' % friction)
    axes[0].invert_yaxis()
    axes[0].set_ylabel('depth [m]')
    for ax, (_, xlabel) in zip(axes, panels):
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=8, loc='best')
    fig.suptitle('%s: fault dip profile at along strike %.0f m (friction %g, cohesion %g bar)'
                 % (os.path.basename(os.path.normpath(case_dir)), strike, friction, cohesion))
    fig.tight_layout()

    if png_filename is None:
        os.makedirs(os.path.join(case_dir, 'fault_plots'), exist_ok=True)
        png_filename = os.path.join(case_dir, 'fault_plots', 'dip_profile_strike%.0fm.png' % strike)
    fig.savefig(png_filename, dpi=150)
    plt.close(fig)
    np.savetxt(os.path.splitext(png_filename)[0] + '.csv', np.vstack(table), delimiter=',', comments='',
               header='step,depth_m,' + ','.join(fields), fmt=['%d'] + ['%.6g'] * (len(fields) + 1))
    if verbose:
        print('dip profile at along strike %.0f m (%d faces, steps %s) -> %s' % (strike, column.size, steps, png_filename))
    return png_filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Append FSP and the Mohr-Coulomb criterion to fault<step>.vtu files.')
    parser.add_argument('case_dir', nargs='?', default=os.path.join('results', 'sol_cpp_single_phase_inj_no_damage_zone'),
                        help='results folder of a finished run')
    parser.add_argument('--friction', type=float, default=0.6, help='friction coefficient')
    parser.add_argument('--cohesion', type=float, default=0.0, help='cohesion [bar]')
    parser.add_argument('--plot', nargs='*', default=['FSP'], metavar='FIELD',
                        help='fields to save as 2D maps per step (default: FSP; no value: no maps)')
    parser.add_argument('--profile', nargs='*', type=float, default=None, metavar='STRIKE',
                        help='save 1D profiles vs depth along the fault dip at the along-strike positions STRIKE [m] '
                             '(no value: at the face with the largest FSP)')
    parser.add_argument('--profile-steps', nargs='+', type=int, default=None, metavar='STEP',
                        help='report steps drawn in the dip profiles (default: first, middle, last)')
    args = parser.parse_args()
    postprocess_case(args.case_dir, friction=args.friction, cohesion=args.cohesion)
    if args.plot:
        plot_fault_case(args.case_dir, fields=args.plot)
    if args.profile is not None:
        for strike in args.profile or [None]:
            plot_fault_dip_profiles(args.case_dir, strike=strike, steps=args.profile_steps, friction=args.friction,
                                    cohesion=args.cohesion)
