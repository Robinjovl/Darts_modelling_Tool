"""Driver for the 1D elastic wave test (thesis Sec. 6.4.1).

Usage:
    python main.py [--scheme backward_euler|newmark|generalized_alpha|hht|bathe] [--nz 100] [--cfl 0.706]
                   [--t_end 0.012] [--snapshots 0.002,0.012] [--params '{"rho_inf": 0.5}'] [--out sol_dir]

Regression entry point (models/run_test_suite2.py convention): run_test(args, platform) with
args = [case, overwrite]; see REGRESSION_CASES and README.md.
"""

import argparse
import json
import os
import sys
import time as wall

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from darts.engines import redirect_darts_output, timer_node
from darts.tools.gen_msh import generate_box_3d

from model import DAY, Model

try:
    from darts.engines import set_num_threads

    set_num_threads(1)
except Exception:
    pass


def generate_mesh(nz: int, H: float = 10.0, a: float = 1.0, mesh_dir: str = None, nxy: int = 1) -> str:
    mesh_dir = mesh_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    filename = os.path.join(mesh_dir, f'column_{nz}.msh' if nxy == 1 else f'column_{nz}_{nxy}x{nxy}.msh')
    if not os.path.isfile(filename):
        tags = {'BND_X-': 991, 'BND_X+': 992, 'BND_Y-': 993, 'BND_Y+': 994, 'BND_Z-': 995, 'BND_Z+': 996,
                'MATRIX': 99991}
        generate_box_3d(X=a, Y=a, Z=H, NX=nxy, NY=nxy, NZ=nz, filename=filename, tags=tags,
                        is_transfinite=True, is_recombine=True, refinement_mult=1)
    return filename


SCHEMES = ('backward_euler', 'newmark', 'generalized_alpha', 'hht', 'bathe')


def apply_scheme(engine, scheme: str, params: dict):
    """Configure the engine's time integration of the inertia term.

    scheme: 'backward_euler' (legacy 3-point BE), 'newmark' (params gamma, beta),
            'generalized_alpha' (param rho_inf), 'hht' (param alpha in [-1/3, 0]),
            'bathe' (param bathe_gamma, default 0.5; two engine sub-steps per timestep).
    Common optional param: kv_damping (Kelvin-Voigt artificial viscosity q, eta = q dt).
    Same semantics as darts.nonlinear_solvers.configure_time_integration (kept local so that
    the driver's defaults, e.g. hht alpha = -0.1, are explicit).
    """
    from darts.engines import time_integration as TI

    if scheme not in SCHEMES:
        raise ValueError(f'unknown scheme {scheme}; choose from {SCHEMES}')
    engine.reset_dynamic_state()
    if scheme == 'backward_euler':
        engine.time_integration = TI.BACKWARD_EULER
    elif scheme == 'newmark':
        engine.set_newmark(params.get('gamma', 0.5), params.get('beta', 0.25))
    elif scheme == 'generalized_alpha':
        engine.set_generalized_alpha(params.get('rho_inf', 0.8))
    elif scheme == 'hht':
        engine.set_hht_alpha(params.get('alpha', -0.1))
    elif scheme == 'bathe':
        engine.time_integration = TI.BATHE
        engine.bathe_gamma = params.get('bathe_gamma', 0.5)
        engine.bathe_substep = 1
    engine.kv_damping = params.get('kv_damping', 0.0)


def layer_average(z, values, decimals=6):
    """Average cell values over the cells sharing the same z (column cross-section); returns (z_layers, mean)."""
    zr = np.round(z, decimals)
    zu, inv = np.unique(zr, return_inverse=True)
    mean = np.zeros(zu.size)
    np.add.at(mean, inv, values)
    counts = np.bincount(inv)
    return zu, mean / counts


def set_top_bc(m, dt_days: float, t_new_days: float):
    """Advance the boundary data over the (sub-)step [t_new - dt, t_new] (both in days).

    reservoir.update() copies bc_rhs (level n) into bc_rhs_prev BEFORE set_top_displacement writes
    the level-(n+1) value; update_trans() then pushes both into mesh.bc / mesh.bc_prev.
    NOTE for a timestep retry (not implemented; the driver aborts on a failed step): do NOT call
    reservoir.update() again (it would overwrite bc_rhs_prev with the failed level-(n+1) value);
    only set_top_displacement(u_top(t_n + dt_new)) + update_trans() are needed.
    """
    m.reservoir.update(dt=dt_days, time=t_new_days)
    m.reservoir.set_top_displacement(m.u_top(t_new_days * DAY))
    m.reservoir.update_trans(dt_days, m.physics.engine.X)


def run(scheme='backward_euler', nz=100, cfl=0.706, t_end=0.012, snapshots=(0.002, 0.012), out_dir=None,
        scheme_params=None, verbose=False, dt_sec=None, first_step_equilibrium=True, nxy=2, side_bc='roller',
        t_pulse=None, u_top=None):
    """nxy: cells across the column; a 1x1 column makes the pm_discretizer roller stencils degenerate
    (u_x/u_y transmissibilities ~1e30), so the default is 2x2 (identical 1D physics, well conditioned).
    t_pulse / u_top: optional pulse length [s] and pulse shape u_top(model, t_sec) replacing the rectangular
    pulse of the thesis (used by convergence_study.py for a smooth raised-cosine pulse)."""
    scheme_params = scheme_params or {}
    mesh = generate_mesh(nz, nxy=nxy)
    m = Model(mesh_filename=mesh, side_bc=side_bc, **({'t_pulse': t_pulse} if t_pulse is not None else {}))
    if u_top is not None:
        import types

        m.u_top = types.MethodType(u_top, m)
    m.init()
    out_dir = out_dir or f'sol_{scheme}_nz{nz}'
    os.makedirs(out_dir, exist_ok=True)
    m.output_directory = out_dir
    redirect_darts_output(os.path.join(out_dir, 'log.txt'))
    m.timer.node["update"] = timer_node()

    e = m.physics.engine
    e.find_equilibrium = True
    e.scale_rows = True
    e.scale_dimless = False

    # trivial equilibrium step (zero initial state and zero BC: bit-identical results without it;
    # kept to exercise the standard quasi-static -> dynamic switch: X = Xn = Xn1 = 0, fluxes_n = 0)
    if first_step_equilibrium:
        m.reservoir.set_equilibrium()
        m.ts_control.dt_first = 1.0
        m.reservoir.update(dt=1.0, time=0.0)
        m.reservoir.update_trans(1.0, e.X)
        conv = m.nonlinear_solver.run_timestep(1.0, 0.0)
        assert conv, 'equilibrium step did not converge'
        # same as THMCModel.reinit() without the VTK snapshot / contact setup
        m.reservoir.turn_off_equilibrium(True)
        eps_vol_ref = np.array(m.reservoir.mesh.ref_eps_vol, copy=False)
        eps_vol_ref[:] = np.asarray(e.eps_vol)[:]
    e.dt1 = 0.0
    e.find_equilibrium = False
    e.t = 0.0

    # pure elasticity: no flow, no Biot coupling
    m.reservoir.apply_geomechanics_mode(physics=m.physics, mode=2)

    # dynamics on
    e.momentum_inertia = m.rho
    apply_scheme(e, scheme, scheme_params)

    dz = m.reservoir.H / nz
    dt_s = dt_sec if dt_sec is not None else cfl * dz / m.c_p  # seconds; the engine is driven in days (dts / DAY)
    nsteps = int(round(t_end / dt_s))
    snap_steps = {int(round(s / dt_s)): s for s in snapshots}
    print(f'scheme={scheme} nz={nz} nxy={nxy} dz={dz:.4g} m c_p={m.c_p:.2f} m/s dt={dt_s:.3e} s CFL={m.c_p*dt_s/dz:.3f} '
          f'nsteps={nsteps}')

    z_cells = m.reservoir.z_centers
    n_matrix = m.reservoir.n_matrix
    z, _ = layer_average(z_cells, np.zeros(n_matrix))
    _, layer_of_cell = np.unique(np.round(z_cells, 6), return_inverse=True)
    results = {'z': z.tolist(), 'nxy': nxy, 'side_bc': side_bc, 'dt_s': dt_s, 'dz': dz, 'c_p': m.c_p, 'scheme': scheme, 'params': scheme_params,
               'snapshots': {}, 'errors': {}, 'newton_iters': [], 'time_series': {'t': [], 'u_top_cell': [],
                                                                                    'u_bottom_cell': []}}
    # Bathe composite: two engine sub-steps per timestep (trapezoidal over gamma*dt, 3-point backward over the
    # rest). MechanicsNewtonSolver.run_timestep performs the split itself and calls on_substep() before each
    # sub-step, where the top displacement is set at the sub-step end time. An older solver without the hook
    # gets the split from the driver (two run_timestep calls with engine.bathe_substep = 1, 2).
    bathe = scheme == 'bathe'
    bathe_gamma = scheme_params.get('bathe_gamma', 0.5) if bathe else 1.0
    solver_splits = bathe and hasattr(m.nonlinear_solver, 'on_substep')
    if solver_splits:
        m.nonlinear_solver.on_substep = lambda substep, dt_sub, t_sub: set_top_bc(m, dt_sub, t_sub + dt_sub)
        substeps = [(1, dt_s)]
    elif bathe:
        substeps = [(1, bathe_gamma * dt_s), (2, (1.0 - bathe_gamma) * dt_s)]
    else:
        substeps = [(1, dt_s)]
    results['bathe_split'] = 'solver' if solver_splits else ('driver' if bathe else None)

    t_sec = 0.0
    t0 = wall.time()
    for step in range(1, nsteps + 1):
        t_new = step * dt_s
        t_sub = t_sec
        n_newton = 0
        for sub, dts in substeps:
            if bathe and not solver_splits:
                e.bathe_substep = sub
            t_sub_new = t_sub + dts
            if not solver_splits:
                set_top_bc(m, dts / DAY, t_sub_new / DAY)
            converged = m.nonlinear_solver.run_timestep(dts / DAY, t_sub / DAY)
            if not converged:
                break
            t_sub = t_sub_new
            n_newton += m.nonlinear_solver.status.n_newton
        if not converged:
            # A retry would need: no reservoir.update() (keeps bc_rhs_prev at level n), a new top displacement
            # at t_n + dt_new; for Bathe a failed 2nd sub-step leaves the engine committed at t_n + gamma*dt
            # (Xn rotated, vel/acc/fluxes_n advanced, a^n lost) -> retry only the 2nd sub-step, or save
            # Xn/vel/acc/vel_n1/fluxes_n before the 1st sub-step and restore them.
            print(f'step {step}: NOT converged, aborting')
            results['failed_step'] = step
            break
        t_sec = t_new
        results['newton_iters'].append(n_newton)
        X = np.asarray(e.X)
        uz_cells = X[2:4 * n_matrix:4]
        _, uz = layer_average(z_cells, uz_cells)
        # 1D check: spread of u_z across the cells of one layer (should be ~roundoff)
        nonuni = float(np.max(np.abs(uz_cells - uz[layer_of_cell])))
        results['time_series']['t'].append(t_sec)
        results['time_series']['u_top_cell'].append(float(uz[np.argmax(z)]))
        results['time_series']['u_bottom_cell'].append(float(uz[np.argmin(z)]))
        if step in snap_steps:
            s = snap_steps[step]
            ua = m.analytical_uz(z, t_sec)
            err_l2 = float(np.sqrt(np.sum((uz - ua) ** 2) * dz))
            err_linf = float(np.max(np.abs(uz - ua)))
            overshoot = float(max(0.0, np.max(np.abs(uz)) - abs(m.u_pulse)))
            # wake: spurious motion left behind the pulse (region already traversed, 10 cells behind the tail);
            # empty (wake = 0) while the pulse tail is less than 10 cells below the top, e.g. at t = 0.002 s
            tail = m.reservoir.H - m.c_p * max(t_sec - m.t_pulse, 0.0)
            wake_mask = z > tail + 10.0 * dz
            wake = float(np.max(np.abs(uz[wake_mask]))) if np.any(wake_mask) else 0.0
            results['snapshots'][f'{s:g}'] = {'t': t_sec, 'uz': uz.tolist(), 'uz_analytical': ua.tolist()}
            results['errors'][f'{s:g}'] = {'l2': err_l2, 'linf': err_linf, 'overshoot': overshoot, 'wake': wake,
                                            'l2_rel': err_l2 / float(np.sqrt(np.sum(ua ** 2) * dz)),
                                            'layer_nonuniformity': nonuni}
            print(f'  t={t_sec:.4g}s  L2={err_l2:.4e}  Linf={err_linf:.4e}  overshoot={overshoot:.4e}  wake={wake:.4e}')
        if verbose and step % 10 == 0:
            print(f'  step {step}/{nsteps} t={t_sec:.4e}s NI={m.nonlinear_solver.status.n_newton}')
    results['wall_time'] = wall.time() - t0
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    print(f'done in {results["wall_time"]:.1f}s; results -> {os.path.join(out_dir, "results.json")}')
    return results


def plot(results, out_png):
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    z = np.array(results['z'])
    order = np.argsort(z)
    snaps = results['snapshots']
    if not snaps:
        print('no snapshots to plot')
        return
    fig, axes = plt.subplots(1, len(snaps), figsize=(6 * len(snaps), 4.5), squeeze=False)
    for ax, snap in zip(axes[0], snaps.values(), strict=True):
        ax.plot(np.array(snap['uz_analytical'])[order], z[order], 'k-', lw=1.5, label='analytical')
        ax.plot(np.array(snap['uz'])[order], z[order], 'r.-', lw=1, ms=3, label=results['scheme'])
        ax.set_xlabel('u_z, m')
        ax.set_ylabel('z, m')
        ax.set_title(f"t = {snap['t']:.4g} s")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ regression
# short cases for models/run_test_suite2.py: nz = 50 (dz = 0.2 m), CFL 0.706 (dt = 2e-4 s), 30 steps to 0.006 s
REGRESSION_CASES = {
    'backward_euler': ('backward_euler', {}),
    'generalized_alpha_0.5': ('generalized_alpha', {'rho_inf': 0.5}),
    'bathe': ('bathe', {'bathe_gamma': 0.5}),
}
REGRESSION_NZ = 50
REGRESSION_T_END = 0.006


def run_test(args: list = None, platform='cpu'):
    """Regression entry point: args = [case, overwrite] (case in REGRESSION_CASES or 'all'; overwrite '1'
    rewrites the reference). Compares the layer-averaged u_z profile at t_end with ref/<case>.npz.
    Returns (fail_flag, simulation_time); (0, 0.0) when a reference was (re)written."""
    args = list(args or [])
    case = str(args[0]) if args else 'all'
    if case not in tuple(REGRESSION_CASES) + ('all',):
        print(f'run_test: unknown case {case!r}; choose from {list(REGRESSION_CASES)} or "all"')
        return 1, 0.0
    overwrite = str(args[-1]) == '1' if len(args) > 1 else False
    cases = list(REGRESSION_CASES) if case == 'all' else [case]
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref')
    os.makedirs(ref_dir, exist_ok=True)
    failed = 0
    saved = False
    sim_time = 0.0
    for name in cases:
        scheme, params = REGRESSION_CASES[name]
        res = run(scheme=scheme, nz=REGRESSION_NZ, cfl=0.706, t_end=REGRESSION_T_END, snapshots=(REGRESSION_T_END,),
                  out_dir=f'sol_test_{name}', scheme_params=params)
        sim_time += res['wall_time']
        key = f'{REGRESSION_T_END:g}'
        if 'failed_step' in res or key not in res['snapshots']:
            print(f'{name}: run failed at step {res.get("failed_step")}')
            failed += 1
            continue
        z = np.array(res['z'])
        uz = np.array(res['snapshots'][key]['uz'])
        ref_file = os.path.join(ref_dir, f'{name}.npz')
        if overwrite or not os.path.isfile(ref_file):
            np.savez(ref_file, z=z, uz=uz, t=res['snapshots'][key]['t'], dt_s=res['dt_s'],
                     newton_iters=np.array(res['newton_iters']))
            print(f'{name}: reference written to {ref_file}')
            saved = True
            continue
        ref = np.load(ref_file)
        ok = (z.shape == ref['z'].shape and np.allclose(z, ref['z'], rtol=1e-8, atol=1e-10)
              and np.allclose(uz, ref['uz'], rtol=1e-8, atol=1e-10))
        err = np.max(np.abs(uz - ref['uz'])) if z.shape == ref['z'].shape else np.inf
        print(f'{name}: {"OK" if ok else "FAIL"}  max|uz - ref| = {err:.3e}  errors = {res["errors"][key]}')
        failed += int(not ok)
    if failed:
        return 1, sim_time
    return (0, 0.0) if saved else (0, sim_time)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', default='backward_euler')
    ap.add_argument('--nz', type=int, default=100)
    ap.add_argument('--cfl', type=float, default=0.706)
    ap.add_argument('--dt', type=float, default=None, help='time step in seconds (overrides --cfl)')
    ap.add_argument('--t_end', type=float, default=0.012)
    ap.add_argument('--snapshots', default='0.002,0.012')
    ap.add_argument('--out', default=None)
    ap.add_argument('--params', default='{}', help='JSON dict of engine scheme parameters')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--nxy', type=int, default=2)
    ap.add_argument('--side_bc', default='roller')
    ap.add_argument('--test', nargs='*', default=None, help='regression mode: [case] [overwrite]')
    args = ap.parse_args()
    if args.test is not None:
        fail, t_sim = run_test(args.test)
        print(f'run_test -> fail={fail} sim_time={t_sim:.2f}s')
        sys.exit(fail)
    snaps = tuple(float(s) for s in args.snapshots.split(','))
    res = run(scheme=args.scheme, nz=args.nz, cfl=args.cfl, t_end=args.t_end, snapshots=snaps, out_dir=args.out,
              scheme_params=json.loads(args.params), verbose=args.verbose, dt_sec=args.dt, nxy=args.nxy,
              side_bc=args.side_bc)
    out_dir = args.out or f'sol_{args.scheme}_nz{args.nz}'
    plot(res, os.path.join(out_dir, 'profiles.png'))
