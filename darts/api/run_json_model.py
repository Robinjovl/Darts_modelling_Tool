"""
CLI entry point for running a DARTS model from a JSON configuration file.

Parses command-line arguments, loads and validates the JSON spec, builds
the model via ModelBuilder, and executes the simulation.  Intended as the
primary way to run a model non-interactively (e.g. ``python -m
darts.api.run_json_model model.json``).
"""

import argparse
import json
import os
import sys

from darts.api import ModelBuilder, ModelSpec
from darts.api.json_model import JsonModel
from darts.api.presets import resolve_section_presets
from darts.tools.cli import get_darts_path, get_lib_search_var, get_lib_var


def _prepare_env():
    # Prefer search path over forced preload to avoid ABI conflicts
    lib_search_var = get_lib_search_var()
    if lib_search_var:
        os.environ[lib_search_var] = (
            os.environ.get(lib_search_var, "") + os.pathsep + str(get_darts_path())
        )

    # Optional opt-in to force-preload libstdc++.so.6 if absolutely required
    if os.environ.get("DARTS_FORCE_PRELOAD_LIBSTDCXX", "0") in ("1", "true", "True"):
        lib_var = get_lib_var()
        if lib_var:
            os.environ[lib_var] = str(get_darts_path() / "libstdc++.so.6") + (
                ":" + os.environ.get(lib_var, "") if os.environ.get(lib_var, "") else ""
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default=None)
    parser.add_argument('--days', type=float, default=None)
    parser.add_argument('--report-days', type=float, default=None)
    parser.add_argument('--vtk-output', action='store_true', default=False)
    args = parser.parse_args()

    _prepare_env()

    with open(args.json) as fp:
        spec_dict = json.load(fp)

    try:
        spec_dict = resolve_section_presets(spec_dict)
    except Exception as e:
        print('Failed to resolve section presets:', e)
        sys.exit(1)

    try:
        spec = ModelSpec.model_validate(spec_dict)
    except Exception as e:
        print('Invalid ModelSpec:', e)
        sys.exit(1)

    m = JsonModel()
    ModelBuilder.apply(spec, m, base_path=os.path.dirname(args.json))
    m.init(platform='cpu')
    # Configure output based on spec if provided, or fallback to defaults
    out_spec = getattr(m, '_output_spec', None)
    if out_spec is not None:
        folder = (
            out_spec.folder
            if getattr(out_spec, 'folder', None) is not None
            else 'output'
        )
        precision = (
            out_spec.precision
            if getattr(out_spec, 'precision', None) is not None
            else 'd'
        )
        m.set_output(output_folder=folder, precision=precision)
    else:
        m.set_output()
    if args.days is None:
        m.run()
        m.print_stat()
        m.print_timers()
        sys.exit(0)

    def _can_write_vtk(model):
        res = getattr(model, 'reservoir', None)
        if res is None:
            return False
        ndims = getattr(res, 'ndims', None)
        if ndims is not None:
            return ndims >= 2
        # Fallback: infer from nx, ny, nz
        nx = getattr(res, 'nx', 1)
        ny = getattr(res, 'ny', 1)
        nz = getattr(res, 'nz', 1)
        dims = int(nx > 1) + int(ny > 1) + int(nz > 1)
        return dims >= 2

    ith_step = 0
    if args.vtk_output and _can_write_vtk(m):
        m.output.output_to_vtk(ith_step=ith_step)
    report_days = args.report_days if args.report_days is not None else args.days
    end_time = m.physics.engine.t + args.days
    while m.physics.engine.t < end_time - 1e-12:
        dt = min(report_days, end_time - m.physics.engine.t)
        m.run(days=dt)
        if args.vtk_output and _can_write_vtk(m):
            m.output.output_to_vtk(ith_step=ith_step + 1)
        ith_step += 1
    m.print_stat()
    m.print_timers()


if __name__ == '__main__':
    main()
