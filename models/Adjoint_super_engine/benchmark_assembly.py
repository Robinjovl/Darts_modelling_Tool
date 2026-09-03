"""Benchmark the device (GPU) vs host adjoint assembly at scale.

Set the grid via ADJ_NX/ADJ_NY/ADJ_NZ. Runs the forward once, then times the
adjoint gradient (grad_adjoint_method_all -> calc_adjoint_gradient_dirac_all)
with engine.adjoint_assembly_on_gpu True vs False, reading the engine timer to
isolate the assembly node from the rest of the backward pass. Skips the slow
numerical-gradient FD loop.

Usage:  ADJ_NX=80 ADJ_NY=80 ADJ_NZ=20 python benchmark_assembly.py
"""

import os
import time
from pathlib import Path

import numpy as np

model_dir = Path(__file__).resolve().parent
os.chdir(model_dir)

from darts.engines import redirect_darts_output

redirect_darts_output("benchmark_assembly.log")

import adjoint_definition as adjoint

adjoint.platform = "gpu"
adjoint.adjoint_solver = "cpra-gpu"
adjoint.use_adjoint_mgr_solver = False

import scipy.optimize as sopt

_real_minimize = sopt.minimize


def _timer_seconds(model, leaf):
    # sum the leaf node's cumulative time wherever it appears under simulation
    sim = model.timer.node["simulation"]
    total = 0.0
    if leaf in sim.node:
        total += sim.node[leaf].get_timer()
    return total


def _bench_minimize(fun, x0, *args, **kwargs):
    jac = kwargs.get("jac", None)
    if jac is None or not callable(jac):
        return _real_minimize(fun, x0, *args, **kwargs)

    model = adjoint._probe_model
    engine = model.physics.engine
    nx, ny, nz = model.reservoir.nx, model.reservoir.ny, model.reservoir.nz
    ncells = nx * ny * nz
    print("BENCH: grid %dx%dx%d = %d cells, %d control vars"
          % (nx, ny, nz, ncells, np.size(x0)), flush=True)

    fun(x0)  # populate history

    def run(on_gpu, reps):
        engine.adjoint_assembly_on_gpu = on_gpu
        leaf = "adjoint jacobian assembly" if on_gpu else "adjoint jacobian assembly host loops"
        jac(x0)  # warm up
        t_asm0 = _timer_seconds(model, leaf)
        t0 = time.time()
        for _ in range(reps):
            jac(x0)
        wall = (time.time() - t0) / reps
        t_asm = (_timer_seconds(model, leaf) - t_asm0) / reps
        return wall, t_asm

    reps = 5
    w_dev, a_dev = run(True, reps)
    w_host, a_host = run(False, reps)

    # correctness cross-check at this scale
    engine.adjoint_assembly_on_gpu = True
    g_dev = np.array(jac(x0), copy=True)
    engine.adjoint_assembly_on_gpu = False
    g_host = np.array(jac(x0), copy=True)
    rel = np.linalg.norm(g_dev - g_host) / (np.linalg.norm(g_host) or 1.0)

    print("BENCH: assembly-only per gradient  device=%.6f s  host=%.6f s  speedup=%.2fx"
          % (a_dev, a_host, (a_host / a_dev) if a_dev else float('nan')), flush=True)
    print("BENCH: whole grad_adjoint per call device=%.4f s  host=%.4f s"
          % (w_dev, w_host), flush=True)
    print("BENCH: device-vs-host gradient rel diff = %.3e" % rel, flush=True)
    raise SystemExit(0)  # skip the numerical-gradient FD loop


sopt.minimize = _bench_minimize
adjoint.minimize = _bench_minimize

import model_definition as md

_orig_init = md.Model.init


def _init_capture(self, *a, **k):
    r = _orig_init(self, *a, **k)
    adjoint._probe_model = self
    return r


md.Model.init = _init_capture

adjoint.prepare_synthetic_observation_data()
adjoint.read_observation_data()
adjoint.process_adjoint()
