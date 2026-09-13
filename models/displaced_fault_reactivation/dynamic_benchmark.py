"""Mixed quasi-static -> dynamic benchmark of the displaced-fault model for the inertia time-integration schemes.

Runs the 'mixed' well-depletion configuration (coarse mesh) until the rupture is resolved for a given number of
dynamic steps and records per-step diagnostics of the fault response, so that the schemes
(backward Euler, Newmark, generalized-alpha/HHT, Bathe, Kelvin-Voigt damping) can be compared.

Usage:
    python dynamic_benchmark.py --scheme generalized_alpha --params '{"rho_inf": 0.5, "kv_damping": 0.1}' \
        --max_dynamic_steps 300 --out bench_ga05_kv01
"""

import argparse
import json
import os
import sys
import time as wall

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from darts.engines import set_num_threads
    set_num_threads(1)
except Exception:
    pass

from darts.engines import redirect_darts_output, timer_node  # noqa: E402
from model import Model  # noqa: E402

G_PA = 6.5e9  # shear modulus of the displaced-fault setup [Pa] (E = 2 G (1 + nu), nu = 0.15)


class Diagnostics:
    """Per accepted step: time, dt, Newton/linear counts, slip-area ratio, max slip, max slip rate,
    seismic-moment proxy G * sum(area * slip), shear-traction roughness (total variation along the fault)."""

    def __init__(self, m):
        self.m = m
        self.rows = []
        self.prev_slip = None
        self.t_prev = None
        ud = m.reservoir.unstr_discr
        # local fault-cell index = global cell id - number of matrix cells (the indexing of get_fault_props)
        n_frac = ud.frac_cells_tot
        self.areas = np.zeros(n_frac)
        centers = np.zeros((n_frac, 3))
        for cell_id, cell in ud.frac_cell_info_dict.items():
            loc = cell_id - ud.mat_cells_tot
            self.areas[loc] = ud.faces[cell_id][4].area
            centers[loc] = cell.centroid
        # order fault cells along the fault (by the depth coordinate of the fault-cell centroids)
        self.order = np.argsort(centers[:, 1])
        self.t_wall = wall.time()

    def __call__(self, m, t, dt, dynamic):
        engine = m.physics.engine
        fault = m.reservoir.get_fault_props(np.asarray(engine.X), m.ith_step, engine)
        g = fault['g_local']  # (n_frac, 3): normal, tangential 1, tangential 2 in the local basis
        f = fault['f_local']
        slip = np.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2)
        slip_rate = 0.0
        if self.prev_slip is not None and dt > 0:
            slip_rate = float(np.max(np.abs(slip - self.prev_slip)) / (dt * 86400.0))
        self.prev_slip = slip.copy()
        shear = np.sqrt(f[:, 1] ** 2 + f[:, 2] ** 2)[self.order]
        row = {
            'step': int(m.ith_step),
            't_days': float(t),
            'dt_s': float(dt * 86400.0),
            'dynamic': bool(dynamic),
            'n_newton': int(m.nonlinear_solver.status.n_newton),
            'n_linear': int(m.nonlinear_solver.status.n_linear),
            'slip_area_ratio': float(m.reservoir.calc_slip_areas(engine=engine)[0]),
            'max_slip_m': float(np.max(slip)),
            'max_slip_rate_m_s': slip_rate,
            'moment_Nm_per_m': float(G_PA * np.sum(self.areas * slip)),  # per unit thickness (2D-like setup)
            'shear_traction_tv_bar': float(np.sum(np.abs(np.diff(shear)))),
            'shear_traction_max_bar': float(np.max(shear)),
            'wall_s': wall.time() - self.t_wall,
        }
        self.rows.append(row)
        if dynamic and m.n_dynamic_steps % 10 == 0:
            print('  [dyn %d] slip_area=%.4f max_slip=%.3e m slip_rate=%.3e m/s TV=%.3e bar NI=%d'
                  % (m.n_dynamic_steps, row['slip_area_ratio'], row['max_slip_m'], row['max_slip_rate_m_s'],
                     row['shear_traction_tv_bar'], row['n_newton']))


def run_benchmark(scheme='backward_euler', params=None, max_dynamic_steps=300, out_dir=None, vtk_every=25,
                  mesh_file='meshes/new_setup_coarse.geo', cache_discretizer=True):
    params = params or {}
    config = {'mode': 'mixed',
              'timesteps': 5 * np.ones(4),
              'depletion': {'mode': 'well', 'value': -250.0},
              'friction_law': 'slip_weakening',
              'mesh_file': mesh_file,
              'cache_discretizer': cache_discretizer,
              'time_integration': {'scheme': scheme, **params}}
    import main as drv  # the model driver (run_python, get_output_folder)

    m = Model(config=config)
    m.init()
    m.output_directory = out_dir or ('bench_' + drv.get_output_folder(config))
    os.makedirs(m.output_directory, exist_ok=True)
    redirect_darts_output(os.path.join(m.output_directory, 'log.txt'))
    m.timer.node["update"] = timer_node()
    m.ith_step = 0
    m.max_dynamic_steps = max_dynamic_steps
    m.vtk_every_dynamic = vtk_every
    m.max_steps = 10 ** 9  # the dynamic-step cap controls the length instead of the 1000-step exit

    # fault cell size for the slip-area gating (as in run_and_plot)
    m.min_area = 1.e10
    for contact in m.physics.engine.contacts:
        cell_ids = np.array(contact.cell_ids, copy=True)
        for i in range(cell_ids.size):
            m.min_area = min(m.min_area, m.reservoir.unstr_discr.faces[cell_ids[i]][4].area)
    m.min_area /= np.max(m.reservoir.unstr_discr.mesh_data.points[:, 2]) - np.min(m.reservoir.unstr_discr.mesh_data.points[:, 2])
    m.cut_off_gap_residual = 0.01

    # equilibrium initialization (as in run_and_plot)
    m.reservoir.set_equilibrium()
    m.physics.engine.find_equilibrium = True
    m.physics.engine.scale_rows = True
    m.physics.engine.scale_dimless = False
    m.ts_control.dt_first = 1.0
    drv.run_python(m, 1.0, init_step=True)
    m.reinit(zero_conduction=True)
    m.physics.engine.dt1 = 0.0
    m.physics.engine.find_equilibrium = False
    m.reservoir.apply_geomechanics_mode(physics=m.physics, mode=0)  # well depletion: flow persists

    diag = Diagnostics(m)
    m.step_callback = diag
    m.physics.engine.t = 0.0
    t0 = wall.time()
    for dt in config['timesteps']:
        m.ts_control.dt_max = dt
        m.ts_control.dt_mult = 10.0
        drv.run_python(m, dt)
        if getattr(m, 'stop_requested', False):
            break
    stats = m.nonlinear_solver.stats
    summary = {'scheme': scheme, 'params': params, 'max_dynamic_steps': max_dynamic_steps,
               'n_dynamic_steps': getattr(m, 'n_dynamic_steps', 0), 'wall_time_s': wall.time() - t0,
               'n_be_fallbacks': getattr(m, 'n_be_fallbacks', 0),
               'n_timesteps_total': stats.n_timesteps_total, 'n_timesteps_wasted': stats.n_timesteps_wasted,
               'n_newton_total': stats.n_newton_total, 'n_newton_wasted': stats.n_newton_wasted,
               'n_linear_total': stats.n_linear_total, 'rows': diag.rows}
    with open(os.path.join(m.output_directory, 'dynamic_benchmark.json'), 'w') as fp:
        json.dump(summary, fp)
    print('benchmark done: %d dynamic steps in %.0f s -> %s' % (summary['n_dynamic_steps'], summary['wall_time_s'],
                                                                 m.output_directory))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', default='backward_euler')
    ap.add_argument('--params', default='{}')
    ap.add_argument('--max_dynamic_steps', type=int, default=300)
    ap.add_argument('--out', default=None)
    ap.add_argument('--vtk_every', type=int, default=25)
    ap.add_argument('--mesh', default='meshes/new_setup_coarse.geo')
    args = ap.parse_args()
    run_benchmark(scheme=args.scheme, params=json.loads(args.params), max_dynamic_steps=args.max_dynamic_steps,
                  out_dir=args.out, vtk_every=args.vtk_every, mesh_file=args.mesh)
