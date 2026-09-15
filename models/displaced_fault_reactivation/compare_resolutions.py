"""Compare the co-seismic rupture of two (or more) full runs (run_full.py) at different mesh resolutions.

Reads run_full.json (per-step diagnostics) and the fault VTK snapshots of each run and produces, in --out:
  * rupture_propagation.png: slipping-area fraction, maximum slip and maximum slip rate against the time since
    the dynamic stage started, plus the rupture front extent along the fault (top-most / bottom-most slipping
    cell) against time;
  * slip_profiles.png: slip and slip-rate profiles along the fault at a set of times since nucleation;
  * rupture_comparison.mp4 (or .gif without ffmpeg): side-by-side animation of the slip and slip-rate profiles of
    the runs at matched times.

The per-step diagnostics in run_full.json and the series derived from the fault snapshots do not measure the
same thing: the engine uses the contact state and the instantaneous velocity, the derived series a finite
difference over the snapshot interval, which is ~27% lower at the peak slip rate (116k run: 9.19 m/s against
6.72 m/s). Mixing the two across the runs of one comparison is therefore not like-for-like -- pass --derive_all
to put every run on the derived series, and always do so when any run is still going (and so has no
run_full.json of its own).

usage: python compare_resolutions.py --runs fine=<out_dir> ultra=<out_dir> [--out compare_fine_ultra]
       [--times 0.05,0.1,0.15,0.2,0.3,0.4] [--fps 10] [--derive_all]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import find_ffmpeg, read_pvd, read_vtk  # noqa: E402

DAY = 86400.0


# A quasi-static step is days long, a dynamic one is at most 5e-4 s = 5.8e-9 days; anything below this gap
# between consecutive fault snapshots therefore belongs to the co-seismic stage.
DYN_GAP_DAYS = 1.e-4


def derive_rows_from_vtk(snaps, gap_days=DYN_GAP_DAYS):
    """Time series (slipping-area fraction, max slip, max slip rate) from the fault snapshots alone.

    Used when run_full.json is absent -- a run that is still going, or one that was killed before it could
    write its summary. The dynamic stage is identified by the snapshot spacing, the co-seismic slip is measured
    from the last pre-nucleation snapshot, and the slipping area is the fraction of fault cells whose slip rate
    exceeds SLIP_RATE_THRESHOLD (the diagnostics in run_full.json instead use the engine's contact state, so the
    two differ slightly in absolute value while following the same history)."""
    if len(snaps) < 2:
        return [], None, None
    gaps = [snaps[i + 1][0] - snaps[i][0] for i in range(len(snaps) - 1)]
    dyn_idx = [i + 1 for i, g in enumerate(gaps) if 0 < g < gap_days]
    if not dyn_idx:
        return [], None, None
    first = min(dyn_idx)
    t0 = snaps[first - 1][0]                     # last snapshot before the co-seismic stage
    ref_depth, ref_slip, _, _ = fault_profile(snaps[first - 1][1])
    rows, prev = [], (snaps[first - 1][0], ref_slip)
    for t_days, f in snaps[first:]:
        if t_days - prev[0] > gap_days and rows:
            break                                 # back to quasi-static stepping: co-seismic stage is over
        depth, slip, _, _ = fault_profile(f)
        dt_s = max((t_days - prev[0]) * DAY, 1e-30)
        rate = np.abs(slip - prev[1]) / dt_s
        rows.append({'t_days': t_days, 'dt_s': dt_s, 'dynamic': True,
                     't_dyn_s': (t_days - t0) * DAY,
                     'slip_area_ratio': float((rate > SLIP_RATE_THRESHOLD).mean()),
                     'max_slip_m': float(np.abs(slip - np.interp(depth, ref_depth, ref_slip)).max()),
                     'max_slip_rate_m_s': float(rate.max())})
        prev = (t_days, slip)
    return rows, t0, (rows[-1]['t_days'] if rows else None)


def load_run(out_dir, derive=False):
    """Load a finished run from run_full.json, or an in-flight/killed one from its VTK snapshots alone.

    derive=True forces the VTK-derived series even when run_full.json is there. Mixing the two paths across the
    runs of one comparison is not like-for-like: the engine diagnostics use the contact state and the
    instantaneous velocity, the derived path a finite difference over the snapshot interval, which is ~27% lower
    at the peak slip rate (measured on the 116k run, 9.19 m/s engine against 6.72 m/s derived). Whenever any run
    of a comparison lacks run_full.json -- one still going, say -- every run must be derived."""
    times, files = read_pvd(os.path.join(out_dir, 'solution_fault.pvd'))
    snaps = [(t, os.path.join(out_dir, f)) for t, f in zip(times, files, strict=True)]
    jpath = os.path.join(out_dir, 'run_full.json')
    summary = None
    if os.path.exists(jpath):
        with open(jpath) as fp:
            summary = json.load(fp)
    if summary is not None and not derive:
        rows = [r for r in summary['rows'] if r['dynamic']]
        t0 = rows[0]['t_days'] - rows[0]['dt_s'] / DAY if rows else None
        t_end = rows[-1]['t_days'] if rows else None
        for r in rows:
            r['t_dyn_s'] = (r['t_days'] - t0) * DAY
    else:
        rows, t0, t_end = derive_rows_from_vtk(snaps)
        n_fault = len(fault_profile(snaps[0][1])[0]) if snaps else 0
        if summary is None:
            summary = {'n_matrix': None, 'n_fault_cells': n_fault, 'derived_from_vtk': True,
                       'n_snapshots': len(snaps)}
            print('%s: no run_full.json (run in progress or interrupted) -- time series derived from %d fault '
                  'snapshots, %d of them co-seismic' % (out_dir, len(snaps), len(rows)))
        else:
            summary = {k: v for k, v in summary.items() if k != 'rows'}
            summary.update({'n_fault_cells': n_fault, 'derived_from_vtk': True, 'n_snapshots': len(snaps)})
            print('%s: run_full.json present but the VTK-derived series was requested -- %d of %d fault '
                  'snapshots are co-seismic' % (out_dir, len(rows), len(snaps)))
    # snapshots of the dynamic stage only (the post-arrest quasi-static ones are excluded)
    dyn_snaps = [(t, f) for t, f in snaps if t0 is not None and t0 <= t <= (t_end or t0) + 1e-12]
    return {'summary': summary, 'rows': rows, 't0_days': t0, 't_end_days': t_end, 'snaps': snaps,
            'dyn_snaps': dyn_snaps, 'dir': out_dir}


def fault_profile(filename, dt_s=None):
    """(depth coordinate [m], slip [m], slip rate [m/s] if dt given) sorted along the fault."""
    c, data, _, _ = read_vtk(filename=filename, props=['g_local', 'f_local', 'mu', 'p'])
    ids = np.argsort(c[:, 1])
    depth = 2250 - c[ids, 1]
    slip = data['g_local'][0][ids, 1]
    return depth, slip, data['f_local'][0][ids], data['mu'][0][ids]


SLIP_RATE_THRESHOLD = 1e-2  # m/s: the actively rupturing part of the fault (creep is ~1e-3 m/s, co-seismic 1-10 m/s)


def front_extent(depth, slip_rate, threshold=SLIP_RATE_THRESHOLD):
    s = slip_rate > threshold
    if not s.any():
        return np.nan, np.nan
    return float(depth[s].min()), float(depth[s].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='name=<run_full output dir>')
    ap.add_argument('--out', default='compare_resolutions')
    ap.add_argument('--times', default='0.05,0.1,0.15,0.2,0.3,0.4', help='times since nucleation [s] for the profiles')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--max_frames', type=int, default=300)
    ap.add_argument('--derive_all', action='store_true',
                    help='force the VTK-derived series for every run. Use it whenever any run of the comparison '
                         'is still going: the engine diagnostics and the derived series measure the slip rate '
                         'differently (~27%% apart at the peak), so mixing them across runs is not like-for-like')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = {}
    for spec in args.runs:
        name, d = spec.split('=', 1)
        runs[name] = load_run(d, derive=args.derive_all)
    # A run that has not nucleated yet has no co-seismic snapshot at all: it would contribute an empty time
    # series and, being the shortest, would be chosen as the animation reference below and reduce it to zero
    # frames (an empty video). Drop such a run, and say so, rather than silently produce nothing.
    for name in [n for n, r in runs.items() if not r['dyn_snaps']]:
        print('%s: still quasi-static, no co-seismic snapshot yet -- excluded from the comparison' % name)
        del runs[name]
    if not runs:
        raise SystemExit('none of the given runs has reached the co-seismic stage yet')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    # --- time series
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for name, run in runs.items():
        t = np.array([r['t_dyn_s'] for r in run['rows']])
        nm = run['summary'].get('n_matrix')
        lbl = '%s (%s cells)' % (name, '{:,}'.format(nm)) if nm else \
              '%s (%s fault cells, partial)' % (name, '{:,}'.format(run['summary'].get('n_fault_cells', 0)))
        ax[0, 0].plot(t, [r['slip_area_ratio'] for r in run['rows']], label=lbl)
        ax[0, 1].plot(t, [r['max_slip_m'] for r in run['rows']], label=name)
        ax[1, 0].semilogy(t, np.maximum([r['max_slip_rate_m_s'] for r in run['rows']], 1e-6), label=name)
        # rupture front from consecutive fault snapshots of the dynamic stage (slip rate above threshold)
        tt, top, bot = [], [], []
        prev = None
        for t_days, f in run['dyn_snaps']:
            depth, slip, _, _ = fault_profile(f)
            if prev is not None and t_days > prev[0]:
                rate = np.abs(slip - prev[1]) / ((t_days - prev[0]) * DAY)
                lo, hi = front_extent(depth, rate)
                tt.append((t_days - run['t0_days']) * DAY)
                top.append(lo)
                bot.append(hi)
            prev = (t_days, slip)
        ax[1, 1].plot(tt, top, label='%s top' % name)
        ax[1, 1].plot(tt, bot, '--', label='%s bottom' % name)
    ax[0, 0].set_ylabel('slipping area fraction')
    ax[0, 1].set_ylabel('max slip [m]')
    ax[1, 0].set_ylabel('max slip rate [m/s]')
    ax[1, 1].set_ylabel('rupture extent (slip rate > %g m/s), depth [m]' % SLIP_RATE_THRESHOLD)
    ax[1, 1].invert_yaxis()
    for a in ax.ravel():
        a.set_xlabel('time since nucleation [s]')
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'rupture_propagation.png'), dpi=130)
    plt.close(fig)

    # --- profiles at given times
    times = [float(x) for x in args.times.split(',')]
    fig, ax = plt.subplots(1, len(times), figsize=(3.2 * len(times), 7), sharey=True)
    for k, ts in enumerate(times):
        for name, run in runs.items():
            f = snapshot_at(run, ts)
            if f is None:
                continue
            depth, slip, _, _ = fault_profile(f)
            ax[k].plot(1e3 * np.abs(slip - slip_ref(run, depth)), depth, label=name)
        ax[k].set_title('t = %.3f s' % ts)
        ax[k].set_xlabel('co-seismic slip [mm]')
        ax[k].grid(True, alpha=0.3)
    ax[0].set_ylabel('depth [m]')
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'slip_profiles.png'), dpi=130)
    plt.close(fig)

    # --- animation: profiles of all runs at the times of the run with the fewest dynamic snapshots
    ref = min(runs.values(), key=lambda r: len(r['dyn_snaps']))
    frame_times = [(s[0] - ref['t0_days']) * DAY for s in ref['dyn_snaps']]
    stride = max(1, len(frame_times) // args.max_frames)
    frame_times = frame_times[::stride]
    import matplotlib.animation as animation
    from matplotlib import rcParams
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        rcParams['animation.ffmpeg_path'] = ffmpeg
    fig, ax = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
    lines = {}
    for name in runs:
        lines[name] = (ax[0].plot([], [], label=name)[0], ax[1].plot([], [], label=name)[0])
    ax[0].set_xlabel('co-seismic slip [mm]')
    ax[1].set_xlabel('slip rate [m/s]')
    ax[0].set_ylabel('depth [m]')
    for a in ax:
        a.grid(True, alpha=0.3)
        a.legend(fontsize=9)
    title = fig.suptitle('')
    depth_all = np.concatenate([fault_profile(run['snaps'][0][1])[0] for run in runs.values()])
    ax[0].set_ylim(depth_all.max(), depth_all.min())

    def frame(i):
        ts = frame_times[i]
        smax, vmax = 1e-3, 1e-3
        for name, run in runs.items():
            f, f_prev, dts = snapshot_pair(run, ts)
            if f is None:
                continue
            depth, slip, _, _ = fault_profile(f)
            s = 1e3 * np.abs(slip - slip_ref(run, depth))
            lines[name][0].set_data(s, depth)
            smax = max(smax, np.abs(s).max())
            if f_prev is not None and dts > 0:
                _, slip_prev, _, _ = fault_profile(f_prev)
                v = np.abs(slip - slip_prev) / dts
                lines[name][1].set_data(v, depth)
                vmax = max(vmax, v.max())
        ax[0].set_xlim(-0.05 * smax, 1.1 * smax)
        ax[1].set_xlim(0, 1.1 * vmax)
        title.set_text('t = %.4f s since nucleation' % ts)
        return [ln for pair in lines.values() for ln in pair]

    anim = animation.FuncAnimation(fig, frame, frames=len(frame_times), interval=int(1000 / args.fps))
    if ffmpeg:
        out = os.path.join(args.out, 'rupture_comparison.mp4')
        anim.save(out, writer=animation.FFMpegWriter(fps=args.fps))
    else:
        out = os.path.join(args.out, 'rupture_comparison.gif')
        anim.save(out, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print('written', os.path.join(args.out, 'rupture_propagation.png'), os.path.join(args.out, 'slip_profiles.png'), out,
          '(%d frames)' % len(frame_times))


_slip_ref_cache = {}


def slip_ref(run, depth):
    """Slip at the last quasi-static snapshot before nucleation (the co-seismic slip is measured from it)."""
    key = run['dir']
    if key not in _slip_ref_cache:
        pre = [f for t, f in run['snaps'] if run['t0_days'] is not None and t < run['t0_days']]
        if pre:
            d, s, _, _ = fault_profile(pre[-1])
            _slip_ref_cache[key] = (d, s)
        else:
            _slip_ref_cache[key] = None
    ref = _slip_ref_cache[key]
    if ref is None:
        return np.zeros_like(depth)
    return np.interp(depth, ref[0], ref[1])


def snapshot_at(run, t_since_nucleation_s):
    if run['t0_days'] is None:
        return None
    target = run['t0_days'] + t_since_nucleation_s / DAY
    dyn = run['dyn_snaps']
    if not dyn:
        return None
    t, f = min(dyn, key=lambda s: abs(s[0] - target))
    return f


def snapshot_pair(run, t_since_nucleation_s):
    """Snapshot closest to the time and the preceding one (for the slip rate), with their time difference [s]."""
    if run['t0_days'] is None:
        return None, None, 0.0
    target = run['t0_days'] + t_since_nucleation_s / DAY
    dyn = run['dyn_snaps']
    if not dyn:
        return None, None, 0.0
    k = int(np.argmin([abs(s[0] - target) for s in dyn]))
    if k == 0:
        return dyn[0][1], None, 0.0
    return dyn[k][1], dyn[k - 1][1], (dyn[k][0] - dyn[k - 1][0]) * DAY


if __name__ == '__main__':
    main()
