"""
Fault post-processing utilities for the THM-vs-geomech-proxy model.

Reads the `*_fault.msh` companion mesh produced by
`gen_fault_msh_no_damage_zone.py` (the fault surfaces there carry the physical
tag FAULT = 9991), extracts the fault-face centers and the fault normal, then
evaluates the Mohr-Coulomb slip criterion and the Fault Slip Potential (FSP)
for a given stress / pore-pressure state.

The stress/pore-pressure helpers mirror the reference implementation in
`models-for-induced-seismicity/lab_scale/lab_experiment_models/fault.py`.

Run standalone to evaluate an analytic in-situ (gravity) stress state on the
generated fault:

    python fault.py
"""

import os
import numpy as np
import meshio

FRAC_TAG = 9991  # physical tag assigned to the fault surfaces in *_fault.msh


# ---------------------------------------------------------------------------
# Tensor / traction helpers (Voigt notation: xx, yy, zz, yz, xz, xy)
# ---------------------------------------------------------------------------
def get_tensor_from_voight(t):
    """Return the 3x3 symmetric stress tensor from a 6x1 Voigt array."""
    res = np.zeros((3, 3))
    res[0, 0] = t[0]
    res[1, 1] = t[1]
    res[2, 2] = t[2]
    res[1, 2] = res[2, 1] = t[3]  # yz
    res[0, 2] = res[2, 0] = t[4]  # xz
    res[0, 1] = res[1, 0] = t[5]  # xy
    return res


def get_stress_on_fault(stress_tensor, fault_normal):
    """Normal and shear stress magnitude on a plane with the given normal."""
    fault_normal_T = np.transpose(fault_normal)
    stress_n = stress_tensor @ fault_normal @ fault_normal
    stress_t = np.linalg.norm(
        (np.eye(3, 3) - np.tensordot(fault_normal, fault_normal_T, axes=0))
        @ stress_tensor @ fault_normal)
    return [stress_n, stress_t]


def get_stress_on_fault_from_traction(fault_traction, fault_normal):
    """Normal and shear stress magnitude from a traction vector on the fault."""
    fault_normal_T = np.transpose(fault_normal)
    stress_n = fault_traction @ fault_normal
    stress_t = np.linalg.norm(
        fault_traction @ (np.eye(3, 3) - np.tensordot(fault_normal, fault_normal_T, axes=0)))
    return [stress_n, stress_t]


def compute_normal(point_1, point_2, point_3):
    """Unit normal of the plane through three points (cross product)."""
    x0, y0, z0 = point_1
    x1, y1, z1 = point_2
    x2, y2, z2 = point_3
    ux, uy, uz = [x1 - x0, y1 - y0, z1 - z0]
    vx, vy, vz = [x2 - x0, y2 - y0, z2 - z0]
    cross = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
    normal = np.array(cross, dtype=float)
    normal /= np.linalg.norm(normal)
    return normal


# ---------------------------------------------------------------------------
# Mohr-Coulomb slip criterion and FSP (Fault Slip Potential)
# ---------------------------------------------------------------------------
def mohr_coulomb(fault_normal, stress_total_fault, pore_pressure_fault, friction, cohesion=0.):
    """
    Evaluate the Mohr-Coulomb slip criterion per fault face.

    Parameters
    ----------
    fault_normal : (3,) unit normal of the fault plane.
    stress_total_fault : (n, 6) total stress per fault face, Voigt notation.
    pore_pressure_fault : (n,) pore pressure per fault face (same units as stress).
    friction : friction coefficient (mu).
    cohesion : cohesion (same units as stress).

    Returns
    -------
    mcc : (n,) slip criterion = |tau| - (cohesion + mu * sigma_n_eff).
          mcc > 0  => the face is past the failure line (slips).
    st  : (n,) FSP / stress ratio = |tau| / sigma_n_eff (shear capacity use).
    stress_n, stress_t : (n,) normal and shear stress magnitudes.
    """
    n_fault_points = stress_total_fault.shape[0]
    mcc = np.zeros(n_fault_points)
    st = np.zeros(n_fault_points)
    stress_n = np.zeros(n_fault_points)
    stress_t = np.zeros(n_fault_points)

    for i in range(n_fault_points):
        stress_total_fault_tensor = get_tensor_from_voight(stress_total_fault[i])
        stress_n[i], stress_t[i] = get_stress_on_fault(stress_total_fault_tensor, fault_normal)

        # stress_n taken positive (compression)
        total_eff_stress = np.abs(stress_n[i]) - pore_pressure_fault[i]
        mcc_n_part = cohesion + friction * total_eff_stress
        mcc[i] = np.abs(stress_t[i]) - mcc_n_part
        st[i] = 0. if total_eff_stress == 0 else np.abs(stress_t[i]) / total_eff_stress

    return mcc, st, stress_n, stress_t


def mohr_coulomb_from_traction(fault_normal, fault_traction, pore_pressure_fault, friction, cohesion=0.):
    """Same as `mohr_coulomb` but from a per-face traction vector (n, 3)."""
    n_fault_points = fault_traction.shape[0]
    mcc = np.zeros(n_fault_points)
    st = np.zeros(n_fault_points)
    stress_n = np.zeros(n_fault_points)
    stress_t = np.zeros(n_fault_points)

    for i in range(n_fault_points):
        stress_n[i], stress_t[i] = get_stress_on_fault_from_traction(fault_traction[i], fault_normal)

        total_eff_stress = np.abs(stress_n[i]) - pore_pressure_fault[i]
        mcc_n_part = cohesion + friction * total_eff_stress
        mcc[i] = np.abs(stress_t[i]) - mcc_n_part
        st[i] = 0. if total_eff_stress == 0 else np.abs(stress_t[i]) / total_eff_stress

    return mcc, st, stress_n, stress_t


# ---------------------------------------------------------------------------
# Fault geometry extraction from the *_fault.msh companion mesh
# ---------------------------------------------------------------------------
def read_fault_faces(mesh_filename, frac_tag=FRAC_TAG):
    """
    Read fault-face centers and the fault normal from a `*_fault.msh` mesh.

    `mesh_filename` may be the base mesh (`mesh.msh`) or the fault mesh
    (`mesh_fault.msh`); the `_fault` suffix is added if missing.

    Returns
    -------
    centers : (n, 3) fault-face centroids.
    fault_normal : (3,) unit fault normal.
    areas : (n,) fault-face areas [m^2].
    """
    base, ext = os.path.splitext(mesh_filename)
    if not base.endswith('_fault'):
        mesh_filename = base + '_fault' + ext

    msh = meshio.read(mesh_filename)
    p = msh.points

    cx, cy, cz, v0, ar = [], [], [], [], []
    # mixed meshes may carry both triangle and quad surface blocks
    for block, block_tags in zip(msh.cells, msh.cell_data['gmsh:physical']):
        if block.type not in ('triangle', 'quad'):
            continue
        faces = block.data[np.asarray(block_tags) == frac_tag]
        if len(faces):
            cx.append(p[faces, 0].mean(axis=1))
            cy.append(p[faces, 1].mean(axis=1))
            cz.append(p[faces, 2].mean(axis=1))
            v0.append(faces[:, 0])
            ar.append(_polygon_areas(p[faces]))

    assert cx, 'no fault faces with tag %d found in %s' % (frac_tag, mesh_filename)
    centers = np.column_stack([np.concatenate(cx), np.concatenate(cy), np.concatenate(cz)])
    areas = np.concatenate(ar)

    # fault normal from three well-separated face vertices
    frac_face_v0 = np.concatenate(v0)
    pts = p[frac_face_v0]
    fault_normal = compute_normal(pts[0], pts[pts.shape[0] // 3], pts[-1])
    return centers, fault_normal, areas


def _polygon_areas(faces_xyz):
    """Areas of planar polygons (n, k, 3) via the fan/cross-product formula."""
    v0 = faces_xyz[:, 0:1, :]
    fan = faces_xyz - v0                       # (n, k, 3)
    cross = np.cross(fan[:, 1:-1, :], fan[:, 2:, :])  # triangles (v0,vi,vi+1)
    return 0.5 * np.linalg.norm(cross.sum(axis=1), axis=1)


# ---------------------------------------------------------------------------
# Analytic in-situ (gravity) stress state — for standalone evaluation
# ---------------------------------------------------------------------------
def insitu_stress_and_pressure(centers, rho_rock=2500.0, rho_water=1000.0, g=9.81,
                               k_h=0.7, k_H=0.9, dp=0.0):
    """
    Simple depth-dependent in-situ stress and pore pressure at fault faces.

    z is depth (m, positive downward as in the generated mesh, top z=0).
    Vertical stress  Sv = rho_rock * g * z          (lithostatic)
    Horizontal       Sxx = k_H * Sv,  Syy = k_h * Sv (no shear terms)
    Pore pressure    Pp = rho_water * g * z + dp     (hydrostatic + perturbation)

    Units: stress/pressure returned in MPa. `dp` (reservoir pressure change,
    MPa) lets you probe the FSP sensitivity to injection/depletion.

    Returns
    -------
    stress : (n, 6) total stress, Voigt notation, MPa.
    pore_pressure : (n,) MPa.
    """
    z = centers[:, 2]
    Sv = rho_rock * g * z * 1e-6          # Pa -> MPa
    stress = np.zeros((centers.shape[0], 6))
    stress[:, 0] = k_H * Sv               # xx
    stress[:, 1] = k_h * Sv               # yy
    stress[:, 2] = Sv                     # zz
    # shear terms (yz, xz, xy) left at zero
    pore_pressure = rho_water * g * z * 1e-6 + dp
    return stress, pore_pressure


def summarize(mcc, st, s_n, s_t, label=''):
    """Print a compact summary of the Mohr-Coulomb / FSP result."""
    fmt = lambda v: '%9.3f' % v
    slip_frac = (mcc > 0).sum() / mcc.size
    print('--- fault slip summary %s---' % (('(' + label + ') ') if label else ''))
    print('  n fault faces     :', mcc.size)
    print('  MCC  [MPa] min/mean/max:', fmt(mcc.min()), fmt(mcc.mean()), fmt(mcc.max()))
    print('  FSP (|t|/sn_eff)  min/mean/max:', fmt(st.min()), fmt(st.mean()), fmt(st.max()))
    print('  sigma_n [MPa] mean:', fmt(s_n.mean()))
    print('  sigma_t [MPa] mean:', fmt(s_t.mean()))
    print('  slip area fraction:', '%.4f' % slip_frac)


def evaluate_fault(mesh_filename, friction=0.6, cohesion=0.0, dp=0.0,
                   frac_tag=FRAC_TAG, write_vtk=True, verbose=True):
    """
    End-to-end: read the fault mesh, build an analytic in-situ stress state,
    and evaluate the Mohr-Coulomb slip criterion + FSP on every fault face.

    Returns a dict with centers, fault_normal, mcc, st (FSP), stress_n, stress_t.
    """
    centers, fault_normal, _ = read_fault_faces(mesh_filename, frac_tag=frac_tag)
    stress, pore_pressure = insitu_stress_and_pressure(centers, dp=dp)
    mcc, st, s_n, s_t = mohr_coulomb(fault_normal, stress, pore_pressure,
                                     friction=friction, cohesion=cohesion)
    if verbose:
        print('fault normal      :', np.round(fault_normal, 4))
        summarize(mcc, st, s_n, s_t, label='dp=%.1f MPa' % dp)

    if write_vtk:
        try:
            from pyevtk.hl import pointsToVTK
            base = os.path.splitext(mesh_filename)[0]
            if base.endswith('_fault'):
                base = base[:-len('_fault')]
            out = base + '_fault_fsp'
            pointsToVTK(out,
                        np.ascontiguousarray(centers[:, 0]),
                        np.ascontiguousarray(centers[:, 1]),
                        np.ascontiguousarray(centers[:, 2]),
                        data={'mcc': mcc, 'FSP': st, 'stress_n': s_n, 'stress_t': s_t})
            if verbose:
                print('wrote', out + '.vtu')
        except Exception as e:
            print('[WARN] VTK output skipped:', e)

    return {'centers': centers, 'fault_normal': fault_normal,
            'mcc': mcc, 'st': st, 'stress_n': s_n, 'stress_t': s_t}


# ---------------------------------------------------------------------------
# Post-processing of a finished THM run: fault slip maps per timestep
#
# The THM engine writes results/<case>/solution<step>.vtu with cell-centered
# total stress (`tot_stress`, Voigt, bars) and `pressure` (bars). We map each
# fault face (from mesh_fault.msh) to its nearest matrix cell, build the fault
# TRACTION t = sigma . n from that cell's total stress, and evaluate the
# Mohr-Coulomb slip criterion and FSP with the traction-based routine.
#
# No change to reservoir.py is required: `tot_stress` is already saved to the
# VTU, which is all the traction-based approach needs.
# ---------------------------------------------------------------------------
BARS_TO_MPA = 0.1

# darts Voigt ordering of tot_stress columns: xx, yy, zz, yz, xz, xy
# -> matches get_tensor_from_voight above.


def _cell_centroids(mesh):
    """Centroids of every volume cell in a meshio mesh (mean of its nodes).

    Concatenated in block order, matching the cell_data array layout.
    """
    return np.concatenate([mesh.points[block.data].mean(axis=1) for block in mesh.cells])


def _timestep_of(path):
    base = os.path.splitext(os.path.basename(path))[0]  # 'solution123'
    digits = ''.join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else -1


def _read_pvd_times(case_dir):
    """Map step index -> simulation time [days] from solution.pvd (empty if absent)."""
    import re
    pvd = os.path.join(case_dir, 'solution.pvd')
    times = {}
    if not os.path.exists(pvd):
        return times
    with open(pvd) as fh:
        for line in fh:
            m = re.search(r'timestep="([^"]+)"\s+file="[^"]*?(\d+)\.vtu"', line)
            if m:
                times[int(m.group(2))] = float(m.group(1))
    return times


def _project_to_2d(pts):
    """PCA-project 3D fault points to the two in-plane axes (u=strike, v=dip)."""
    p = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(p, full_matrices=False)
    return p @ Vt[0], p @ Vt[1]


def _plot_fault_scalar(u, v, values, name, tstep, out_dir, unit=''):
    """Contourf map of a per-face scalar on the (PCA-projected) fault plane."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    nu = nv = 200
    ug, vg = np.meshgrid(np.linspace(u.min(), u.max(), nu),
                         np.linspace(v.min(), v.max(), nv))
    zg = griddata((u, v), values, (ug, vg), method='linear')

    vmin = np.nanpercentile(values, 2)
    vmax = np.nanpercentile(values, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    if np.isclose(vmin, vmax):
        c = float(np.nanmean(values)); vmin, vmax = c - 1e-6, c + 1e-6
    levels = np.linspace(vmin, vmax, 21)

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(ug, vg, zg, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=ax, label=name + (f' [{unit}]' if unit else ''))
    ax.set_xlabel('strike u (m)')
    ax.set_ylabel('dip v (m)')
    ax.set_title(f'{name}  timestep={tstep}')
    ax.set_aspect('equal')
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'fault_{name}_tstep{tstep:04d}.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _traction_from_fault_vtu(m):
    """Read (traction (n,3), pore (n,), points (n,3)) from a saved fault*.vtu."""
    data = m.point_data if m.point_data else {k: v[0] for k, v in m.cell_data.items()}
    tr = np.column_stack([np.asarray(data['fault_traction_x']).ravel(),
                          np.asarray(data['fault_traction_y']).ravel(),
                          np.asarray(data['fault_traction_z']).ravel()])
    pore = np.asarray(data['p']).ravel()
    return tr, pore, m.points


def _traction_from_solution_vtu(m, face_cell, fault_normal):
    """Fallback: reconstruct traction t = sigma . n from cell tot_stress."""
    tot_stress = np.concatenate([np.asarray(a) for a in m.cell_data['tot_stress']]) * BARS_TO_MPA
    pressure = np.concatenate([np.asarray(a) for a in m.cell_data['pressure']]) * BARS_TO_MPA
    stress_faces = tot_stress[face_cell]
    traction = np.array([get_tensor_from_voight(s) @ fault_normal for s in stress_faces])
    return traction, pressure[face_cell]


def postprocess_case(case_name, results_root='results',
                     mesh_filename=os.path.join('meshes', 'no_damage_zone', 'mesh_fault.msh'),
                     friction=0.6, cohesion=0.0, frac_tag=FRAC_TAG,
                     step_stride=1, verbose=True):
    """
    Generate per-timestep fault slip maps (FSP and Mohr-Coulomb) for a finished
    THM run in `results/<case_name>/`, using the traction-based criterion.

    Preferred source: the `fault<step>.vtu` files written during the run by
    `reservoir.save_fault_traction` (Hooke-forces traction, as in the lab
    models). If none are present, it falls back to reconstructing the traction
    as sigma.n from the cell `tot_stress` in `solution<step>.vtu`.

    For every timestep it writes two PNGs into `results/<case_name>/fault_plots/`:
      - fault_FSP_tstepNNNN.png  (|t| / sigma_n_eff)
      - fault_mcc_tstepNNNN.png  (slip criterion |t| - mu*sigma_n_eff, MPa)
    """
    from scipy.spatial import cKDTree
    import glob

    case_dir = os.path.join(results_root, case_name)
    out_dir = os.path.join(case_dir, 'fault_plots')
    os.makedirs(out_dir, exist_ok=True)

    centers, fault_normal, areas = read_fault_faces(mesh_filename, frac_tag=frac_tag)
    total_area = areas.sum()
    if verbose:
        print(f'case: {case_name}')
        print(f'fault: {centers.shape[0]} faces, normal {np.round(fault_normal, 4)}, '
              f'total area {total_area:.3e} m2')

    fault_files = sorted(glob.glob(os.path.join(case_dir, 'fault*.vtu')), key=_timestep_of)
    fault_files = [f for f in fault_files if _timestep_of(f) >= 0]
    use_hooke = len(fault_files) > 0

    if use_hooke:
        files = fault_files[::step_stride]
        # fault*.vtu carries its own points (fault-face centers) -> project those
        u, v = _project_to_2d(_traction_from_fault_vtu(meshio.read(files[0]))[2])
        if verbose:
            print(f'source: Hooke-forces fault*.vtu ({len(fault_files)} steps)')
    else:
        files = sorted(glob.glob(os.path.join(case_dir, 'solution*.vtu')), key=_timestep_of)
        files = [f for f in files if _timestep_of(f) >= 0][::step_stride]
        if not files:
            print(f'[WARN] no fault*.vtu or solution*.vtu found in {case_dir}')
            return
        print('[INFO] no fault*.vtu found - falling back to sigma.n from tot_stress. '
              'Re-run the model to save Hooke-forces tractions.')
        u, v = _project_to_2d(centers)
        # map each fault face to nearest matrix cell once (geometry fixed)
        face_cell = cKDTree(_cell_centroids(meshio.read(files[0]))).query(centers)[1]

    slip_series = []  # (tstep, slip_area, slip_area_frac, slip_count_frac, fsp_mean)
    for f in files:
        tstep = _timestep_of(f)
        m = meshio.read(f)
        if use_hooke:
            traction, pore, _ = _traction_from_fault_vtu(m)
        else:
            traction, pore = _traction_from_solution_vtu(m, face_cell, fault_normal)

        mcc, st, s_n, s_t = mohr_coulomb_from_traction(fault_normal, traction, pore,
                                                       friction=friction, cohesion=cohesion)
        _plot_fault_scalar(u, v, st, 'FSP', tstep, out_dir)
        _plot_fault_scalar(u, v, mcc, 'mcc', tstep, out_dir, unit='MPa')

        # slip area: sum of face areas where the Mohr-Coulomb criterion is met
        slipping = mcc > 0
        slip_area = areas[slipping].sum()
        slip_series.append((tstep, slip_area, slip_area / total_area,
                            slipping.mean(), st.mean()))
        if verbose:
            print(f'  tstep {tstep:4d}: FSP mean {st.mean():6.3f}  '
                  f'mcc max {mcc.max():8.3f} MPa  slip area {slip_area:.3e} m2 '
                  f'({100 * slip_area / total_area:5.1f}%)')

    times_days = _read_pvd_times(case_dir)
    _write_slip_series(slip_series, total_area, out_dir, times_days, verbose)
    if verbose:
        print(f'PNGs written to {out_dir}')


def _write_slip_series(slip_series, total_area, out_dir, times_days=None, verbose=True):
    """Write slip-area time series to CSV and a slip-area-vs-time plot
    (absolute [m^2] and normalized [fraction of total fault area])."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arr = np.array(slip_series, dtype=float)  # (n, 5): tstep, area, area_frac, count_frac, fsp
    tsteps = arr[:, 0].astype(int)

    # x-axis: physical time in years if available, else timestep index
    times_days = times_days or {}
    if all(t in times_days for t in tsteps):
        x = np.array([times_days[t] for t in tsteps]) / 365.25
        xlabel = 'time [years]'
    else:
        x = arr[:, 0]
        xlabel = 'timestep'

    csv_path = os.path.join(out_dir, 'slip_area.csv')
    out = np.column_stack([arr[:, 0], x, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]])
    header = 'timestep,time_years,slip_area_m2,slip_area_fraction,slip_count_fraction,fsp_mean'
    np.savetxt(csv_path, out, delimiter=',', header=header, comments='',
               fmt=['%d', '%.6f', '%.6e', '%.6f', '%.6f', '%.6f'])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax1.plot(x, arr[:, 1], '-o', color='C3')
    ax1.set_ylabel('slip area [m$^2$]')
    ax1.set_title(f'Fault slip area vs time  (total fault area {total_area:.3e} m$^2$)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, 100 * arr[:, 2], '-s', color='C0')
    ax2.set_ylabel('slip area fraction [%]')
    ax2.set_xlabel(xlabel)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'slip_area_vs_time.png'), dpi=150)
    plt.close(fig)
    if verbose:
        print(f'slip-area series -> {csv_path} and slip_area_vs_time.png')


if __name__ == '__main__':
    # Post-processing tool: run AFTER the THM model has written its VTUs into
    # results/<case_name>/. Set the case name here.
    case_name = 'sol_cpp_single_phase_thermal_doublet_no_damage_zone'
    postprocess_case(case_name, friction=0.6, cohesion=0.0, step_stride=1)
