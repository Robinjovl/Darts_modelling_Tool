"""Backward Euler on the rectangular pulse (thesis Sec. 6.4.1 / Fig. 6.1b): the observed convergence order depends on
the error measure, not on the scheme.

Joint refinement at CFL 0.706 (dt ~ dz), errors of u_z against the d'Alembert solution at t = 2 ms and 12 ms in
  * the mesh-weighted L2 norm (sum e^2 dz)^1/2 and the L1 norm (sum |e| dz) used by convergence_study.py,
  * the Euclidean norm of the cell-error vector divided by the number of cells, (sum e^2)^1/2 / N, which is the
    quantity whose magnitude and slope reproduce the thesis / ECMOR-2026 figure ("first order with respect to
    sqrt(dz dt)"): it equals the mesh-weighted L2 norm times sqrt(dz) / H, so it adds 1/2 to the apparent order.
The exact solution is discontinuous, so a first-order dissipative scheme smears the jump over a width
~ sqrt(nu t) with numerical diffusion nu ~ dt: the L1 error then scales like dt^1/2 and the mesh-weighted L2 error
like dt^1/4 (references drawn), whatever the formal order of the time integrator.

usage: python backward_euler_norms.py [--nz 50,100,200,400,800] [--cfl 0.706] [--out figures] [--runs_dir DIR]
(--runs_dir reuses existing run directories <runs_dir>/A_be_nz<nz>/results.json, e.g. from an earlier invocation)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as drv  # noqa: E402

TIMES = (0.002, 0.012)


def norms(z, uz, ua):
    dz = z[1] - z[0]
    e = uz - ua
    return {
        'L2': float(np.sqrt(np.sum(e**2) * dz)),
        'L1': float(np.sum(np.abs(e)) * dz),
        'Linf': float(np.abs(e).max()),
        'eucl/N': float(np.sqrt(np.sum(e**2)) / len(e)),
        'amp': float(np.abs(uz).max()),
    }


def orders(h, err):
    h, err = np.array(h), np.array(err)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def collect(nz_list, cfl, runs_dir):
    rows = {}
    for nz in nz_list:
        out_dir = os.path.join(runs_dir, f'A_be_nz{nz}')
        f = glob.glob(os.path.join(out_dir, 'results.json'))
        if f:
            r = json.load(open(f[0]))
        else:
            r = drv.run(scheme='backward_euler', nz=nz, cfl=cfl, t_end=max(TIMES), snapshots=TIMES, out_dir=out_dir)
        z = np.array(r['z'])
        rec = {'dz': r['dz'], 'dt': r['dt_s'], 'sqrt_dzdt': float(np.sqrt(r['dz'] * r['dt_s']))}
        for t in TIMES:
            s = r['snapshots'][f'{t:g}']
            rec[f'{t:g}'] = norms(z, np.array(s['uz']), np.array(s['uz_analytical']))
        rows[nz] = rec
        print(f'nz={nz:4d} dz={rec["dz"]:.4f} dt={rec["dt"]:.3e} '
              + '  '.join(f't={t:g}: L2 {rec[f"{t:g}"]["L2"]:.3e} L1 {rec[f"{t:g}"]["L1"]:.3e} '
                          f'eucl/N {rec[f"{t:g}"]["eucl/N"]:.3e} amp {rec[f"{t:g}"]["amp"]:.4f}' for t in TIMES), flush=True)
    return rows


def write_table(rows, path):
    nzs = sorted(rows)
    lines = ['Backward Euler, rectangular pulse, joint refinement at fixed CFL; errors against the d\'Alembert solution and '
             'observed orders w.r.t. dz (= w.r.t. dt; the order w.r.t. sqrt(dz dt) is twice as large).', '']
    for t in TIMES:
        lines.append(f'### t = {t * 1e3:g} ms')
        lines.append('| norm | ' + ' | '.join(f'nz={nz}' for nz in nzs) + ' | orders | LS-fit order |')
        lines.append('|---|' + '---|' * (len(nzs) + 2))
        h = [rows[nz]['dz'] for nz in nzs]
        for k in ('amp', 'L2', 'L1', 'Linf', 'eucl/N'):
            vals = [rows[nz][f'{t:g}'][k] for nz in nzs]
            if k == 'amp':
                lines.append('| pulse amplitude retained | ' + ' | '.join(f'{v / 0.01:.2f}' for v in vals) + ' | | |')
                continue
            o = orders(h, vals)
            fit = np.polyfit(np.log(h), np.log(vals), 1)[0]
            lines.append(f'| {k} | ' + ' | '.join(f'{v:.3e}' for v in vals) + ' | '
                         + ', '.join(f'{x:.2f}' for x in o) + f' | {fit:.2f} |')
        lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def plot(rows, out_png):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    nzs = sorted(rows)
    x = np.array([rows[nz]['sqrt_dzdt'] for nz in nzs])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = (('L2', 'mesh-weighted L2 error [m sqrt(m)]', ((0.5, 'slope 1/2 in sqrt(dz dt) = dt^1/4'),)),
              ('L1', 'L1 error [m^2]', ((1.0, 'slope 1 in sqrt(dz dt) = dt^1/2'),)),
              ('eucl/N', '|u_z - u_zh|_2 / N  (thesis Fig. 6.1b)', ((2.0, 'slope 2 in sqrt(dz dt) = dt^1'), (1.5, 'slope 3/2 = dt^3/4'))))
    for ax, (k, label, refs) in zip(axes, panels, strict=True):
        for t, style in zip(TIMES, ('o-', 's-'), strict=True):
            y = np.array([rows[nz][f'{t:g}'][k] for nz in nzs])
            ax.loglog(x, y, style, color='tab:blue' if t == TIMES[0] else 'tab:red', label=f't = {t * 1e3:g} ms')
        y0 = np.array([rows[nz][f'{TIMES[1]:g}'][k] for nz in nzs])[-1]
        for slope, name in refs:
            ax.loglog(x, y0 * 0.7 * (x / x[-1]) ** slope, 'k:', lw=1, label=name)
        ax.set_xlabel('sqrt(dz dt) [sqrt(m s)]')
        ax.set_ylabel(label)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title('backward Euler, rectangular pulse, CFL 0.706')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print('written', out_png)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nz', default='50,100,200,400,800')
    ap.add_argument('--cfl', type=float, default=0.706)
    ap.add_argument('--out', default='figures')
    ap.add_argument('--runs_dir', default=None, help='directory with A_be_nz<nz>/results.json (default: <out>/backward_euler_runs)')
    args = ap.parse_args()
    nz_list = [int(x) for x in args.nz.split(',')]
    os.makedirs(args.out, exist_ok=True)
    runs_dir = args.runs_dir or os.path.join(args.out, 'backward_euler_runs')
    rows = collect(nz_list, args.cfl, runs_dir)
    with open(os.path.join(args.out, 'backward_euler_norms.json'), 'w') as f:
        json.dump({'cfl': args.cfl, 'rows': rows}, f, indent=1)
    write_table(rows, os.path.join(args.out, 'backward_euler_norms.md'))
    plot(rows, os.path.join(args.out, 'backward_euler_norms.png'))
