"""Compare dynamic_benchmark.py runs (per-step fault diagnostics) of several time-integration schemes.

Usage: python compare_dynamic_benchmarks.py --runs be=path1 ga05_kv01=path2 ... [--out compare_dir]
"""
import argparse
import json
import os

import numpy as np


def load(path):
    with open(os.path.join(path, 'dynamic_benchmark.json')) as f:
        d = json.load(f)
    rows = [r for r in d['rows'] if r['dynamic']]
    keys = ['t_days', 'dt_s', 'n_newton', 'n_linear', 'slip_area_ratio', 'max_slip_m', 'max_slip_rate_m_s',
            'moment_Nm_per_m', 'shear_traction_tv_bar', 'shear_traction_max_bar', 'wall_s']
    series = {k: np.array([r[k] for r in rows]) for k in keys}
    t0 = series['t_days'][0] - series['dt_s'][0] / 86400.0
    series['t_dyn_s'] = (series['t_days'] - t0) * 86400.0
    return d, series


def summarize(name, d, s):
    dyn_wall = s['wall_s'][-1] - s['wall_s'][0]
    return {
        'run': name, 'scheme': d['scheme'], 'params': json.dumps(d['params']), 'n_dyn': int(len(s['t_dyn_s'])),
        't_dyn_s': float(s['t_dyn_s'][-1]), 'wasted_steps': int(d['n_timesteps_wasted']),
        'NI_mean': float(np.mean(s['n_newton'])), 'LI_mean': float(np.mean(s['n_linear'])),
        'wall_per_step_s': float(dyn_wall / max(len(s['t_dyn_s']) - 1, 1)),
        'slip_area_end': float(s['slip_area_ratio'][-1]), 'max_slip_end_m': float(s['max_slip_m'][-1]),
        'peak_slip_rate_m_s': float(np.max(s['max_slip_rate_m_s'])),
        'moment_end': float(s['moment_Nm_per_m'][-1]),
        'TV_mean_bar': float(np.mean(s['shear_traction_tv_bar'])), 'TV_max_bar': float(np.max(s['shear_traction_tv_bar'])),
        # roughness of the slip-rate history itself: total variation of log10(max slip rate) per step (oscillation indicator)
        'slip_rate_history_tv': float(np.sum(np.abs(np.diff(np.log10(np.maximum(s['max_slip_rate_m_s'][1:], 1e-12)))))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='name=path pairs')
    ap.add_argument('--out', default='compare_dynamic')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = {}
    for item in args.runs:
        name, path = item.split('=', 1)
        runs[name] = load(path)
    rows = [summarize(n, d, s) for n, (d, s) in runs.items()]
    cols = ['run', 'n_dyn', 't_dyn_s', 'wasted_steps', 'NI_mean', 'LI_mean', 'wall_per_step_s', 'slip_area_end',
            'max_slip_end_m', 'peak_slip_rate_m_s', 'moment_end', 'TV_mean_bar', 'TV_max_bar', 'slip_rate_history_tv']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(f'{r[c]:.4g}' if isinstance(r[c], float) else str(r[c]) for c in cols) + ' |')
    table = '\n'.join(lines)
    print(table)
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(table + '\n')
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    panels = [('slip_area_ratio', 'slipping area ratio'), ('max_slip_m', 'max slip [m]'),
              ('max_slip_rate_m_s', 'max slip rate [m/s]'), ('moment_Nm_per_m', 'moment proxy G*sum(A*slip) [N]'),
              ('shear_traction_tv_bar', 'shear traction TV along fault [bar]'), ('n_newton', 'Newton iterations / step')]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, (key, label) in zip(axes.ravel(), panels):
        for name, (d, s) in runs.items():
            ax.plot(s['t_dyn_s'], s[key], lw=1, label=name)
        ax.set_xlabel('time since dynamic switch [s]')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if key == 'max_slip_rate_m_s':
            ax.set_yscale('log')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'time_series.png'), dpi=110)
    print('written', os.path.join(args.out, 'time_series.png'))


if __name__ == '__main__':
    main()
