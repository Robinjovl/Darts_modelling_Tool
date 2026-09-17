"""Validate the device (GPU) adjoint assembly against the host reference.

Runs the forward once, then computes the adjoint gradient twice on the SAME
history -- once with engine.adjoint_assembly_on_gpu = True (device kernel) and
once = False (host reference loop) -- and compares the two gradient vectors.
The host path is already validated to match the CPU adjoint to ~3.8e-8, so a
device-vs-host match to ~1e-10 proves the CUDA port is faithful.

Usage:  python phase1_validate.py
"""

import os
import time
from pathlib import Path

import numpy as np

model_dir = Path(__file__).resolve().parent
os.chdir(model_dir)

from darts.engines import redirect_darts_output

redirect_darts_output("phase1_validate.log")

import adjoint_definition as adjoint

adjoint.platform = "gpu"
adjoint.adjoint_solver = "cpra-gpu"
adjoint.use_adjoint_mgr_solver = False

import scipy.optimize as sopt

_real_minimize = sopt.minimize


def _probe_minimize(fun, x0, *args, **kwargs):
    jac = kwargs.get("jac", None)
    if jac is not None and callable(jac):
        model = adjoint._probe_model
        engine = model.physics.engine

        print("PHASE1: forward once to populate history", flush=True)
        fun(x0)

        # Device assembly
        engine.adjoint_assembly_on_gpu = True
        t0 = time.time()
        g_dev = np.array(jac(x0), copy=True)
        t_dev = time.time() - t0

        # Host assembly (reference)
        engine.adjoint_assembly_on_gpu = False
        t0 = time.time()
        g_host = np.array(jac(x0), copy=True)
        t_host = time.time() - t0

        # A couple of repeats to time steady-state (first call warms up)
        engine.adjoint_assembly_on_gpu = True
        reps = 3
        t0 = time.time()
        for _ in range(reps):
            jac(x0)
        t_dev_avg = (time.time() - t0) / reps
        engine.adjoint_assembly_on_gpu = False
        t0 = time.time()
        for _ in range(reps):
            jac(x0)
        t_host_avg = (time.time() - t0) / reps

        num = np.linalg.norm(g_dev - g_host)
        den = np.linalg.norm(g_host)
        rel = num / den if den else num
        cos = g_dev.dot(g_host) / (np.linalg.norm(g_dev) * np.linalg.norm(g_host))
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        max_abs = np.max(np.abs(g_dev - g_host))
        print("PHASE1: |g_dev|=%.10e |g_host|=%.10e" % (np.linalg.norm(g_dev), den), flush=True)
        print("PHASE1: device-vs-host rel=%.3e  angle=%.3e deg  max_abs_diff=%.3e"
              % (rel, ang, max_abs), flush=True)
        print("PHASE1: gradient timing (whole grad_adjoint_method_all): "
              "device=%.4f s  host=%.4f s  (avg over %d: dev=%.4f host=%.4f)"
              % (t_dev, t_host, reps, t_dev_avg, t_host_avg), flush=True)
        verdict = "PASS" if rel < 1e-9 else ("CLOSE" if rel < 1e-6 else "FAIL")
        print("PHASE1: VERDICT = %s (threshold rel<1e-9)" % verdict, flush=True)
    return _real_minimize(fun, x0, *args, **kwargs)


sopt.minimize = _probe_minimize
adjoint.minimize = _probe_minimize

# stash the proxy_model so the probe can reach the engine; patch process_adjoint's
# Model construction by intercepting after init.
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
