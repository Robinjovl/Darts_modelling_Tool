"""ThreadSanitizer data-race check, run under CHECK_DATA_RACES=1 CI jobs.

Mirrors helper_scripts/valgrind_check.py's structure and conventions -- see
that file for the general pattern this follows (per-model subprocess, log
parsing rather than trusting the tool's own exit code, one summary file per
model). The two differ in what they check (leaks vs. races) and, necessarily,
in how much they can run: ThreadSanitizer's runtime overhead (roughly 5-15x
native, worse for HYPRE's AMG setup) rules out reusing valgrind_check.py's
full model list in any reasonable CI time budget, so this list is
deliberately small and biased toward models that actually exercise the
OpenMP/MT engine + wells path, not toward physics coverage.

Requires a build with -T / ENABLE_TSAN=ON (helper_scripts/build_darts_cmake.sh).
"""

import glob
import importlib
import os
import re
import shutil
import subprocess
import sys
import time

# Deliberately short: see the module docstring. nx=100, 1D, dead-oil-with-wells
# -- small enough to finish under TSan in CI time, while still exercising the
# MT engine's parallel Jacobian assembly and well-block evaluation (the two
# call paths whose interaction produced the interpolator cache race this job
# exists to catch a regression of).
tsan_models = [
    '2ph_do',
]

# if the user passed extra models, append them (comma-separated):
extra = os.environ.get('APPEND_TSAN_MODEL')
if extra:
    for m in extra.split(','):
        m = m.strip()
        if m and m not in tsan_models:
            tsan_models.append(m)

# folder with logs to be reported
log_folder = os.path.join('models', '_tsan_logs')

# ThreadSanitizer suppressions file: known runtime/library false positives,
# not application bugs -- see the file itself for the reasoning behind each
# entry.
suppression_file = 'helper_scripts/tsan.supp'

# how many threads to run each model with: >1 is required for the race to be
# reachable at all (darts.engines defaults to half the machine's cores unless
# OMP_NUM_THREADS is set, so this just pins it to something small and fixed
# for reproducible CI timing rather than relying on the runner's core count).
OMP_NUM_THREADS = os.environ.get('TSAN_OMP_NUM_THREADS', '4')

# How many simulated days to run each model for. Deliberately NOT
# m.data_ts.dt_first (valgrind_check.py's convention, via the shared
# run_model() below): a single tiny first step is enough to exercise most
# alloc/dealloc paths for leak detection, but empirically found to be far too
# short to reliably reach a second call into the same OpenMP-parallelized
# routines (GMRES's Krylov loop, HYPRE's AMG setup) that data races actually
# need -- a real local test run of dt_first alone found zero races and never
# even wrote a TSan log. 30 days (2ph_do's full configured runtime is 300)
# reliably exercises HYPRE/CPR + GMRES + engine init in ~15-45s under TSan.
RUN_DAYS = float(os.environ.get('TSAN_RUN_DAYS', '30'))

RACE_PATTERN = re.compile(r'^WARNING: ThreadSanitizer: data race', re.MULTILINE)


def run_model(model):
    # add it also to system path to load modules
    sys.path.insert(0, os.path.abspath(r'.'))

    # clear any cached bytecode
    shutil.rmtree("__pycache__", ignore_errors=True)

    # import the model definition and run
    try:
        # assume model.py defines a class Model
        mod = importlib.import_module('model')
        m = mod.Model()
        m.init(platform='cpu')
        m.set_output()
        m.run(days=RUN_DAYS, save_well_data=False, save_reservoir_data=False)
        print(f"[OK] {model}")
        success = True
    except Exception as e:
        print(f"[FAIL] {model} -> {e}")
        success = False

    return success


def resolve_libtsan():
    """Find the runtime TSan library to LD_PRELOAD, matching whichever
    compiler actually built it (the CI image's from-source GCC bundles its
    own; a plain system gcc/clang also each carry one). Not hardcoded: the
    exact .so version differs across toolchains and images."""
    for cc in (os.environ.get('CC'), 'gcc', 'cc', 'clang'):
        if not cc or not shutil.which(cc):
            continue
        try:
            path = (
                subprocess.check_output([cc, '-print-file-name=libtsan.so'])
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            continue
        if path and path != 'libtsan.so' and os.path.exists(path):
            return path
    return None


def analyze_log(log_path):
    """Count ThreadSanitizer data-race warnings surviving suppression."""
    try:
        content = open(log_path, errors='replace').read()
    except FileNotFoundError:
        return 0

    return len(RACE_PATTERN.findall(content))


def run_tsan_for_model(model, libtsan, timeout=1800):
    model_path = os.path.join('models', model)
    if not os.path.isdir(model_path):
        print(f"[SKIP] Model directory not found: {model_path}")
        return True  # failed

    abs_log_folder = os.path.abspath(log_folder)
    abs_suppression_file = os.path.abspath(suppression_file)
    os.makedirs(abs_log_folder, exist_ok=True)

    log_path = os.path.join(abs_log_folder, f'{model}.tsan.log')
    prog_out = os.path.join(abs_log_folder, f'{model}.log')
    summary_file = os.path.join(abs_log_folder, f'{model}.summary.txt')

    py_snippet = (
        'import sys, os; '
        'from helper_scripts.tsan_check import run_model; '
        f'model_path = os.path.join("models", "{model}"); '
        'os.chdir(model_path); '
        f'sys.exit(0 if run_model(model="{model}") else 1)'
    )

    cmd = ['darts', '-c', py_snippet]

    print(f'Running ThreadSanitizer for model {model}...')
    starting_time = time.time()

    env = os.environ.copy()
    env['LD_PRELOAD'] = libtsan
    env['OMP_NUM_THREADS'] = OMP_NUM_THREADS
    # exitcode=0: don't let TSan's own exit-code-on-warning decide pass/fail --
    # this script derives that from the (suppression-aware) log content
    # instead, same reasoning as valgrind_check.py's --error-exitcode=0.
    env['TSAN_OPTIONS'] = (
        f'halt_on_error=0:exitcode=0:history_size=7:'
        f'suppressions={abs_suppression_file}:log_path={log_path}'
    )

    with open(prog_out, 'w') as out_f:
        try:
            proc = subprocess.run(
                cmd, stdout=out_f, stderr=subprocess.STDOUT, env=env, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            print(f'ERROR: timeout running TSan for model {model}')
            return True  # failed

    # TSAN_OPTIONS log_path=<path> writes one file per process, suffixed with
    # its PID (<path>.<pid>); merge them the same way valgrind_check.py merges
    # its per-PID logs, so the artifact stays one file per model.
    parts = sorted(glob.glob(f'{log_path}.*'))
    if parts:
        with open(log_path, 'w') as merged:
            for part in parts:
                with open(part, errors='replace') as pf:
                    merged.write(pf.read())
                os.remove(part)

    n_races = analyze_log(log_path)
    with open(summary_file, 'w') as sf:
        sf.write(f'ThreadSanitizer summary for model: {model}\n')
        sf.write(f'Data races (post-suppression): {n_races}\n')

    elapsed = time.time() - starting_time

    if proc.returncode != 0:
        print(
            f'[FAIL] {model} returned exit code {proc.returncode},\t\t{elapsed:.2f} s'
        )
        return True  # failed: model itself errored/crashed
    elif n_races != 0:
        print(
            f'[FAIL] ThreadSanitizer found {n_races} data race(s) for model {model},\t\t{elapsed:.2f} s'
        )
        return True  # failed: found races
    else:
        print(f'[OK] TSan check finished for {model},\t\t{elapsed:.2f} s')
        return False  # success


def main():
    libtsan = resolve_libtsan()
    if not libtsan:
        print(
            'Error: could not locate libtsan.so via any of gcc/cc/clang -print-file-name'
        )
        sys.exit(1)
    print(f'Using TSan runtime: {libtsan}')

    if not os.path.exists(suppression_file):
        print(f'Error: suppression file not found: {suppression_file}')
        sys.exit(1)

    os.makedirs(log_folder, exist_ok=True)

    all_ok = True
    for model in tsan_models:
        result = run_tsan_for_model(model, libtsan)
        if result:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
