"""Full displaced-fault test on one mesh: production from the remote well under quasi-static stepping, the fully
dynamic co-seismic stage once the quasi-static Newton loop fails at nucleation, and the return to quasi-static
stepping after the rupture arrests (main.run_python). Writes the VTK output, the fault-profile animation
(main.plot_profiles) and per-step rupture diagnostics (dynamic_benchmark.Diagnostics) to the output folder.

usage: python run_full.py --mesh meshes/new_setup_fine.geo [--qs_solver cudss --dyn_solver cudss] [--threads 8]
                          [--days 5,5,5,5] [--vtk_every 10] [--scheme backward_euler --params '{}'] [--out DIR]
                          [--fps 10 --stride 1] [--device 0]
"""
import argparse
import json
import os
import sys
import time as wall

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as drv  # noqa: E402
from dynamic_benchmark import Diagnostics, timer_snapshot  # noqa: E402


def run_full(mesh_file, qs_solver='cudss', dyn_solver='cudss', threads=8, days=(5.0, 5.0, 5.0, 5.0), vtk_every=10,
             scheme='backward_euler', params=None, out_dir=None, fps=10, stride=1, cache_discretizer=True,
             depletion=-250.0, min_dynamic_steps=100, arrest_area_fraction=0.005, dt_after_arrest=1.e-3,
             max_dynamic_steps=None):
    config = {'mode': 'mixed',
              'timesteps': list(days),
              'depletion': {'mode': 'well', 'value': depletion},
              'friction_law': 'slip_weakening',
              'mesh_file': mesh_file,
              'cache_discretizer': cache_discretizer,
              'time_integration': {'scheme': scheme, **(params or {})},
              'linear_solver': {'quasi_static': qs_solver, 'dynamic': dyn_solver},
              'n_threads': threads,
              'vtk_every_dynamic': vtk_every,
              'max_steps': 10 ** 9,
              'min_dynamic_steps': min_dynamic_steps,
              'arrest_area_fraction': arrest_area_fraction,
              'dt_after_arrest': dt_after_arrest,
              'animation_fps': fps,
              'animation_stride': stride}
    if max_dynamic_steps:
        # cap the co-seismic stage (the arrest tail is the expensive part); the run still writes the
        # diagnostics, the VTK output and the animation when the cap is reached
        config['max_dynamic_steps'] = int(max_dynamic_steps)
    # run_and_plot builds the model, finds the equilibrium, runs the schedule and makes the animation; the
    # diagnostics are attached through a wrapper of the Model class so run_and_plot stays untouched
    diag_holder = {}
    original_model = drv.Model

    class ModelWithDiagnostics(original_model):
        def init(self, *a, **kw):
            super().init(*a, **kw)
            diag_holder['diag'] = Diagnostics(self)
            self.step_callback = diag_holder['diag']

    drv.Model = ModelWithDiagnostics
    if out_dir is not None:
        drv.get_output_folder = lambda config=None, _o=out_dir: _o
    t0 = wall.time()
    drv.run_and_plot(config=config, plot_analytics=False)
    diag = diag_holder.get('diag')
    m = diag.m if diag is not None else None
    summary = {'mesh_file': mesh_file, 'config': {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in config.items()},
               'wall_time_s': wall.time() - t0}
    if m is not None:
        stats = m.nonlinear_solver.stats
        summary.update({'n_timesteps_total': stats.n_timesteps_total, 'n_timesteps_wasted': stats.n_timesteps_wasted,
                        'n_newton_total': stats.n_newton_total, 'n_linear_total': stats.n_linear_total,
                        'n_arrests': getattr(m, 'n_arrests', 0), 'n_be_fallbacks': getattr(m, 'n_be_fallbacks', 0),
                        'n_dynamic_stalls': getattr(m, 'n_dynamic_stalls', 0),
                        'n_matrix': int(m.reservoir.n_matrix), 'n_blocks': int(m.reservoir.mesh.n_blocks),
                        'timers': timer_snapshot(m.timer), 'rows': diag.rows})
        with open(os.path.join(m.output_directory, 'run_full.json'), 'w') as fp:
            json.dump(summary, fp)
        print('run_full done: %d steps (%d wasted), %d Newton, %d linear, %d arrest(s) (%d stall-triggered), %.0f s -> %s'
              % (stats.n_timesteps_total, stats.n_timesteps_wasted, stats.n_newton_total, stats.n_linear_total,
                 summary['n_arrests'], summary['n_dynamic_stalls'], summary['wall_time_s'], m.output_directory))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', default='meshes/new_setup_fine.geo')
    ap.add_argument('--qs_solver', default='cudss')
    ap.add_argument('--dyn_solver', default='cudss')
    ap.add_argument('--threads', type=int, default=8, help='OpenMP threads for the assembly (bit-identical to 1 thread)')
    ap.add_argument('--days', default='5,5,5,5', help='comma-separated reporting periods [days]')
    ap.add_argument('--vtk_every', type=int, default=10, help='VTK output every n-th dynamic step')
    ap.add_argument('--scheme', default='backward_euler')
    ap.add_argument('--params', default='{}')
    ap.add_argument('--out', default=None)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--stride', type=int, default=1, help='animation frame stride over the fault snapshots')
    ap.add_argument('--no_cache', action='store_true')
    ap.add_argument('--depletion', type=float, default=-250.0)
    ap.add_argument('--min_dynamic_steps', type=int, default=100)
    ap.add_argument('--arrest_area_fraction', type=float, default=0.005)
    ap.add_argument('--dt_after_arrest', type=float, default=1.e-3)
    ap.add_argument('--max_dynamic_steps', type=int, default=None,
                    help='stop after this many dynamic steps (captures rupture propagation without the arrest tail)')
    args = ap.parse_args()
    run_full(args.mesh, args.qs_solver, args.dyn_solver, args.threads, [float(x) for x in args.days.split(',')],
             args.vtk_every, args.scheme, json.loads(args.params), args.out, args.fps, args.stride,
             not args.no_cache, args.depletion, args.min_dynamic_steps, args.arrest_area_fraction, args.dt_after_arrest,
             args.max_dynamic_steps)
