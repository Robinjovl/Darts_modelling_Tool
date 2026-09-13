"""Plot fault profiles (slip, shear/normal traction, friction, yield function) of several runs at one output step.

Usage: python compare_fault_profiles.py --runs name=dir ... --step 300 --out fig.png
"""
import argparse
import os

import meshio
import numpy as np


def read_fault(path, step):
    m = meshio.read(os.path.join(path, f'solution_fault{step}.vtu'))
    pts = m.points
    cells = m.cells[0].data
    centers = pts[cells].mean(axis=1)
    data = {k: v[0] for k, v in m.cell_data.items()}
    return centers, data


def step_at_time(path, t_dyn):
    """Written output step (solution_fault<step>.vtu) closest to the given time since the dynamic switch."""
    import json
    import re
    with open(os.path.join(path, 'dynamic_benchmark.json')) as f:
        rows = [r for r in json.load(f)['rows'] if r['dynamic']]
    t0 = rows[0]['t_days'] - rows[0]['dt_s'] / 86400.0
    written = sorted(int(m.group(1)) for fn in os.listdir(path) for m in [re.match(r'solution_fault(\d+)\.vtu', fn)] if m)
    t_of_step = {r['step']: (r['t_days'] - t0) * 86400.0 for r in rows}
    cand = [(abs(t_of_step[st] - t_dyn), st) for st in written if st in t_of_step]
    return min(cand)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--step', type=int, default=300)
    ap.add_argument('--time', type=float, default=None,
                    help='physical time since the dynamic switch [s]; picks, per run, the written step closest to it')
    ap.add_argument('--out', default='fault_profiles.png')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(22, 6), sharey=True)
    stats = []
    for item in args.runs:
        name, path = item.split('=', 1)
        step = args.step
        if args.time is not None:
            step = step_at_time(path, args.time)
            print(f'{name}: closest written step to t_dyn={args.time:g} s is {step}')
        c, d = read_fault(path, step)
        order = np.argsort(c[:, 1])
        y = c[order, 1]
        slip = np.linalg.norm(d['g_local'][order, 1:], axis=1)
        shear = np.linalg.norm(d['f_local'][order, 1:], axis=1) / 10.0  # bar -> MPa
        normal = -d['f_local'][order, 0] / 10.0
        mu = d['mu'][order]
        phi = d['phi'][order]
        axes[0].plot(slip * 1e3, y, '.-', ms=3, lw=1, label=name)
        axes[1].plot(shear, y, '.-', ms=3, lw=1)
        axes[2].plot(normal, y, '.-', ms=3, lw=1)
        axes[3].plot(mu, y, '.-', ms=3, lw=1)
        axes[4].plot(phi, y, '.-', ms=3, lw=1)
        # oscillation indicator: second differences of shear traction along the fault (high-pass)
        d2 = np.abs(np.diff(shear, 2))
        stats.append((name, float(np.max(slip)), float(np.sum(slip > 1e-6)), float(np.mean(d2)), float(np.max(d2))))
    for ax, xl in zip(axes, ['slip [mm]', 'shear traction [MPa]', 'normal traction [MPa]', 'friction coeff.', 'phi']):
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('y along fault [m]')
    axes[0].legend(fontsize=8)
    fig.suptitle('fault profiles at ' + (f't_dyn = {args.time:g} s' if args.time is not None else f'output step {args.step}'))
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print('| run | max slip [m] | n cells slip>1um | mean |d2 shear| [MPa] | max |d2 shear| [MPa] |')
    print('|---|---|---|---|---|')
    for s in stats:
        print(f'| {s[0]} | {s[1]:.3e} | {s[2]:.0f} | {s[3]:.3e} | {s[4]:.3e} |')


if __name__ == '__main__':
    main()
