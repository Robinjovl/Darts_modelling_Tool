import argparse
import json
import os
import sys

from darts.api import ModelBuilder, ModelSpec
from darts.api.json_model import JsonModel
from darts.tools.cli import get_darts_path, get_lib_var


def _prepare_env():
    lib_var = get_lib_var()
    if not lib_var:
        return
    if lib_var == 'LD_PRELOAD':
        preload_lib = str(get_darts_path() / 'libstdc++.so.6')
        existing = os.environ.get(lib_var, "")
        os.environ[lib_var] = preload_lib + (":" + existing if existing else "")
    else:
        os.environ[lib_var] = (
            str(get_darts_path()) + os.pathsep + os.environ.get(lib_var, "")
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
        # Pydantic v2
        if hasattr(ModelSpec, 'model_validate'):
            spec = ModelSpec.model_validate(spec_dict)
        else:  # Pydantic v1 fallback
            spec = ModelSpec.parse_obj(spec_dict)
    except Exception as e:
        print('Invalid ModelSpec:', e)
        sys.exit(1)

    m = JsonModel()
    ModelBuilder.apply(spec, m)
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
