"""Run the 1D pulse test for a matrix of time-integration schemes and summarise oscillation metrics.

Usage: python compare_schemes.py [--nz 100] [--cfl 0.706] [--out sweep_dir] [--only name1,name2]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as drv  # noqa: E402

CASES = {
    'backward_euler': ('backward_euler', {}),
    'newmark_trapezoidal': ('newmark', {'gamma': 0.5, 'beta': 0.25}),
    'newmark_g0.6': ('newmark', {'gamma': 0.6, 'beta': 0.3025}),
    'hht_-0.05': ('hht', {'alpha': -0.05}),
    'hht_-0.3': ('hht', {'alpha': -0.3}),
    'genalpha_0.8': ('generalized_alpha', {'rho_inf': 0.8}),
    'genalpha_0.5': ('generalized_alpha', {'rho_inf': 0.5}),
    'genalpha_0.0': ('generalized_alpha', {'rho_inf': 0.0}),
    'bathe': ('bathe', {'bathe_gamma': 0.5}),
    'trapezoidal_kv0.25': ('newmark', {'gamma': 0.5, 'beta': 0.25, 'kv_damping': 0.25}),
    'genalpha_0.5_kv0.25': ('generalized_alpha', {'rho_inf': 0.5, 'kv_damping': 0.25}),
    'genalpha_0.5_kv0.1': ('generalized_alpha', {'rho_inf': 0.5, 'kv_damping': 0.1}),
    'bathe_kv0.1': ('bathe', {'bathe_gamma': 0.5, 'kv_damping': 0.1}),
}


def run_sweep(nz, cfl, out_dir, only=None, t_end=0.012, snapshots=(0.002, 0.012), figures_dir='figures'):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    all_results = {}
    for name, (scheme, params) in CASES.items():
        if only and name not in only:
            continue
        print(f'===== {name}: scheme={scheme} params={params}')
        res = drv.run(scheme=scheme, nz=nz, cfl=cfl, t_end=t_end, snapshots=snapshots,
                      out_dir=os.path.join(out_dir, name), scheme_params=params)
        all_results[name] = res
        for key, err in res['errors'].items():
            rows.append({'case': name, 'scheme': scheme, 'params': json.dumps(params), 't': float(key),
                         **err, 'wall_time': res['wall_time'],
                         'mean_newton': float(np.mean(res['newton_iters'])) if res['newton_iters'] else np.nan,
                         'failed_step': res.get('failed_step', 0)})
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    # markdown table
    lines = ['| case | t [s] | L2 rel | Linf | overshoot | wake | NI/step | fail |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['case']} | {r['t']:g} | {r['l2_rel']:.3f} | {r['linf']:.2e} | {r['overshoot']:.2e} | "
                     f"{r['wake']:.2e} | {r['mean_newton']:.2f} | {r['failed_step']} |")
    table = '\n'.join(lines)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(table + '\n')
    print(table)
    plot_all(all_results, os.path.join(out_dir, 'profiles_all.png'))
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        plot_overlay(all_results, os.path.join(figures_dir, 'step_wave_comparison.png'))
        plot_all(all_results, os.path.join(figures_dir, 'step_wave_all_schemes.png'))
        with open(os.path.join(figures_dir, 'step_wave_summary.md'), 'w') as f:
            f.write(table + '\n')
    return rows, all_results


# groups of the overlay figure: (title, [case names])
OVERLAY_GROUPS = [
    ('second-order schemes without stabilizer', ['newmark_trapezoidal', 'hht_-0.05', 'genalpha_0.8', 'genalpha_0.5', 'bathe']),
    ('numerically dissipative schemes', ['backward_euler', 'genalpha_0.0', 'newmark_g0.6', 'hht_-0.3']),
    ('Kelvin-Voigt stabilized (eta = q dt)', ['trapezoidal_kv0.25', 'genalpha_0.5_kv0.1', 'genalpha_0.5_kv0.25', 'bathe_kv0.1']),
]


def plot_overlay(all_results, out_png):
    """All methods against the analytical step wave: one row per scheme family, one column per snapshot."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    if not all_results:
        return
    snaps = list(next(iter(all_results.values()))['snapshots'].keys())
    groups = [(t, [n for n in names if n in all_results]) for t, names in OVERLAY_GROUPS]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(len(groups), len(snaps), figsize=(6.5 * len(snaps), 3.6 * len(groups)), squeeze=False,
                             sharex='col')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for r, (title, names) in enumerate(groups):
        for c, key in enumerate(snaps):
            ax = axes[r, c]
            first = all_results[names[0]]
            z = np.array(first['z'])
            order = np.argsort(z)
            snap = first['snapshots'][key]
            ax.plot(z[order], np.array(snap['uz_analytical'])[order], 'k-', lw=2, label='analytical (d\'Alembert)')
            for k, name in enumerate(names):
                res = all_results[name]
                s = res['snapshots'].get(key)
                if s is None:
                    continue
                e = res['errors'][key]
                ax.plot(np.array(res['z'])[order], np.array(s['uz'])[order], '-', lw=1.1, color=colors[k % len(colors)],
                        label=f"{name} (L2rel {e['l2_rel']:.2f})")
            ax.set_title(f"{title}, t = {snap['t'] * 1e3:.3g} ms", fontsize=10)
            ax.set_ylabel('u_z [m]')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='lower right' if c == 0 else 'lower left')
            if r == len(groups) - 1:
                ax.set_xlabel('z [m]')
    dz = first['dz']
    fig.suptitle(f"1D elastic column, rectangular pulse -0.01 m / 1 ms; nz = {int(round(10.0 / dz))}, "
                 f"CFL = {first['c_p'] * first['dt_s'] / dz:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print('written', out_png)


def plot_all(all_results, out_png):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    names = list(all_results.keys())
    if not names:
        return
    snaps = list(next(iter(all_results.values()))['snapshots'].keys())
    ncol = len(snaps)
    nrow = len(names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 2.6 * nrow), squeeze=False, sharex='col')
    for r, name in enumerate(names):
        res = all_results[name]
        z = np.array(res['z'])
        order = np.argsort(z)
        for c, key in enumerate(snaps):
            ax = axes[r, c]
            snap = res['snapshots'].get(key)
            if snap is None:
                continue
            ax.plot(z[order], np.array(snap['uz_analytical'])[order], 'k-', lw=1.2, label='analytical')
            ax.plot(z[order], np.array(snap['uz'])[order], 'r-', lw=1, label=name)
            err = res['errors'][key]
            ax.set_title(f"{name}  t={snap['t']:.3g}s  L2rel={err['l2_rel']:.2f} over={err['overshoot']:.1e} "
                         f"wake={err['wake']:.1e}", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.0135, 0.0035)
            if r == nrow - 1:
                ax.set_xlabel('z, m')
            ax.set_ylabel('u_z, m')
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nz', type=int, default=100)
    ap.add_argument('--cfl', type=float, default=0.706)
    ap.add_argument('--out', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--t_end', type=float, default=0.012)
    ap.add_argument('--figures', default='figures', help='directory for step_wave_comparison.png (empty: skip)')
    args = ap.parse_args()
    only = args.only.split(',') if args.only else None
    run_sweep(args.nz, args.cfl, args.out or f'sweep_nz{args.nz}_cfl{args.cfl:g}', only=only, t_end=args.t_end,
              figures_dir=args.figures or None)
