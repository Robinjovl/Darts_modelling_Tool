"""Convergence study of the inertia time integrators on the 1D column (thesis Sec. 6.4.1 set-up).

Two pulses are used:
* the rectangular (step) pulse of the thesis, whose exact solution is discontinuous, so every scheme is
  limited to (at most) first order and the error is dominated by the spatial dispersion of the front;
* a smooth raised-cosine (Hann) pulse of the same amplitude, on which the temporal order of the schemes is
  observable (second order for the Newmark / generalized-alpha / HHT / Bathe family, first order for
  backward Euler, Newmark gamma > 1/2 and the Kelvin-Voigt-damped variants).

Joint space-time refinement at a fixed CFL number (dt ~ dz), error in L2 against the d'Alembert solution
at a time before the first reflection. Produces figures/convergence.png and figures/convergence.md.

Usage: python convergence_study.py [--nz 50,100,200,400] [--cfl 0.706] [--only name1,name2] [--out figures]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as drv  # noqa: E402

# schemes: name -> (scheme, params, line style)
SCHEMES = {
    'backward Euler': ('backward_euler', {}, dict(color='k', marker='s')),
    'Newmark trapezoidal': ('newmark', {'gamma': 0.5, 'beta': 0.25}, dict(color='tab:red', marker='o')),
    'Newmark g=0.6': ('newmark', {'gamma': 0.6, 'beta': 0.3025}, dict(color='tab:orange', marker='v')),
    'HHT a=-0.05': ('hht', {'alpha': -0.05}, dict(color='tab:purple', marker='^')),
    'gen-alpha r=0.8': ('generalized_alpha', {'rho_inf': 0.8}, dict(color='tab:blue', marker='D')),
    'gen-alpha r=0.5': ('generalized_alpha', {'rho_inf': 0.5}, dict(color='tab:cyan', marker='d')),
    'Bathe': ('bathe', {'bathe_gamma': 0.5}, dict(color='tab:green', marker='P')),
    'trapezoidal + KV q=0.1': ('newmark', {'gamma': 0.5, 'beta': 0.25, 'kv_damping': 0.1},
                               dict(color='tab:red', marker='o', linestyle='--')),
    'gen-alpha 0.5 + KV q=0.1': ('generalized_alpha', {'rho_inf': 0.5, 'kv_damping': 0.1},
                                 dict(color='tab:cyan', marker='d', linestyle='--')),
    'Bathe + KV q=0.1': ('bathe', {'bathe_gamma': 0.5, 'kv_damping': 0.1},
                         dict(color='tab:green', marker='P', linestyle='--')),
}

PULSES = {
    # rectangular pulse of the thesis: amplitude u0 for 0 < t <= T
    'step': {'t_pulse': 1.e-3, 'u_top': lambda self, t: self.u_pulse if 0.0 < t <= self.t_pulse * (1 + 1e-9) else 0.0},
    # raised-cosine pulse of length T (smooth, C1)
    'smooth': {'t_pulse': 2.e-3,
               'u_top': lambda self, t: 0.5 * self.u_pulse * (1.0 - np.cos(2.0 * np.pi * t / self.t_pulse))
               if 0.0 < t < self.t_pulse else 0.0},
}
T_END = {'step': 0.012, 'smooth': 0.010}  # s, before the first reflection at H / c_p = 14.2 ms


def run_case(pulse, scheme, params, nz, cfl, out_dir):
    """Run one case with the requested pulse shape (patched on the Model class) and return (L2, Linf)."""
    r = drv.run(scheme=scheme, nz=nz, cfl=cfl, t_end=T_END[pulse], snapshots=(T_END[pulse],), out_dir=out_dir,
                scheme_params=params, t_pulse=PULSES[pulse]['t_pulse'], u_top=PULSES[pulse]['u_top'])
    key = next(iter(r['errors']))
    return r['errors'][key]['l2'], r['errors'][key]['linf']


def study(nz_list, cfl, only=None, out_root='figures'):
    os.makedirs(out_root, exist_ok=True)
    runs_dir = os.path.join(out_root, 'convergence_runs')
    results = {}  # pulse -> name -> {'dz': [], 'l2': [], 'linf': []}
    for pulse in PULSES:
        results[pulse] = {}
        for name, (scheme, params, _) in SCHEMES.items():
            if only and name not in only:
                continue
            rec = {'dz': [], 'l2': [], 'linf': []}
            for nz in nz_list:
                tag = f"{pulse}_{name.replace(' ', '_').replace('=', '').replace('+', '')}_nz{nz}"
                l2, linf = run_case(pulse, scheme, params, nz, cfl, os.path.join(runs_dir, tag))
                rec['dz'].append(10.0 / nz)
                rec['l2'].append(l2)
                rec['linf'].append(linf)
                print(f'{pulse:6s} {name:26s} nz={nz:4d}  L2={l2:.4e}  Linf={linf:.4e}', flush=True)
            results[pulse][name] = rec
    with open(os.path.join(out_root, 'convergence.json'), 'w') as f:
        json.dump({'nz': nz_list, 'cfl': cfl, 'results': results}, f, indent=1)
    write_table(results, nz_list, cfl, os.path.join(out_root, 'convergence.md'))
    plot(results, cfl, os.path.join(out_root, 'convergence.png'))
    return results


def orders(dz, err):
    dz, err = np.array(dz), np.array(err)
    return np.log(err[:-1] / err[1:]) / np.log(dz[:-1] / dz[1:])


def write_table(results, nz_list, cfl, path):
    lines = [f'Joint refinement at CFL = {cfl} (dt = CFL dz / c_p); L2 error vs the d\'Alembert solution and the '
             'observed order between successive levels.', '']
    for pulse, res in results.items():
        lines.append(f'### {pulse} pulse')
        lines.append('| scheme | ' + ' | '.join(f'nz={nz}' for nz in nz_list) + ' | orders | LS-fit order |')
        lines.append('|---|' + '---|' * (len(nz_list) + 2))
        for name, rec in res.items():
            p = orders(rec['dz'], rec['l2'])
            fit = np.polyfit(np.log(rec['dz']), np.log(rec['l2']), 1)[0] if len(rec['dz']) > 1 else float('nan')
            lines.append(f'| {name} | ' + ' | '.join(f'{e:.3e}' for e in rec['l2']) + ' | '
                         + ', '.join(f'{x:.2f}' for x in p) + f' | {fit:.2f} |')
        lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def plot(results, cfl, out_png):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    pulses = list(results.keys())
    fig, axes = plt.subplots(1, len(pulses), figsize=(6.5 * len(pulses), 5.2))
    axes = np.atleast_1d(axes)
    for ax, pulse in zip(axes, pulses, strict=True):
        res = results[pulse]
        for name, rec in res.items():
            style = SCHEMES[name][2]
            ax.loglog(rec['dz'], rec['l2'], label=name, lw=1.3, ms=6, **style)
        # reference slopes anchored at the coarsest level of the trapezoidal scheme (or the first scheme)
        ref = res.get('Newmark trapezoidal') or next(iter(res.values()))
        dz0, e0 = ref['dz'][0], ref['l2'][0]
        dz = np.array(sorted(ref['dz']))
        ax.loglog(dz, e0 * (dz / dz0) ** 1, 'k:', lw=1, label='slope 1')
        ax.loglog(dz, e0 * (dz / dz0) ** 2, 'k-.', lw=1, label='slope 2')
        ax.set_xlabel(f'cell size dz [m]   (dt = CFL dz / c_p, CFL = {cfl:g})')
        ax.set_ylabel('L2 error of u_z [m]')
        ax.set_title(f'{pulse} pulse, t = {T_END[pulse] * 1e3:g} ms')
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()
    axes[-1].legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print('written', out_png)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nz', default='50,100,200,400')
    ap.add_argument('--cfl', type=float, default=0.706)
    ap.add_argument('--only', default=None, help='comma-separated subset of scheme names')
    ap.add_argument('--out', default='figures')
    args = ap.parse_args()
    nz_list = [int(x) for x in args.nz.split(',')]
    only = [s.strip() for s in args.only.split(',')] if args.only else None
    study(nz_list, args.cfl, only=only, out_root=args.out)
