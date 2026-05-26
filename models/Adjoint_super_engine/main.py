import os
import argparse
from pathlib import Path

from darts.engines import redirect_darts_output

import adjoint_definition as adjoint

# this adjoint gradient test is based on 2ph_comp model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adjoint-solver",
        choices=("mgr", "superlu"),
        default=None,
        help="Override adjoint solver for this run. Omit to use adjoint_definition.use_adjoint_mgr_solver.",
    )
    parser.add_argument(
        "--log",
        default="run.log",
        help="DARTS log path, relative to this model directory unless absolute.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(__file__).resolve().parent
    os.chdir(model_dir)

    if args.adjoint_solver is not None:
        adjoint.use_adjoint_mgr_solver = args.adjoint_solver == "mgr"

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = model_dir / log_path
    redirect_darts_output(str(log_path))

    solver_name = "MGR" if adjoint.use_adjoint_mgr_solver else "SuperLU"
    print("Adjoint solver mode: %s" % solver_name)

    adjoint.prepare_synthetic_observation_data()
    adjoint.read_observation_data()
    failed = adjoint.process_adjoint()
    print('----------------The status is: %s' % failed)
    return failed


if __name__ == '__main__':
    raise SystemExit(main())
