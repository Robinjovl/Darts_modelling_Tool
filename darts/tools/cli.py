"""
A CLI for DARTS, which ensures that the runtime environment is properly set up for
running DARTS scripts and models.

You can either manually specify the path to python scripts:
    `darts models/2ph_comp/main.py`

Or run a model script via a JSON file:
    `darts --json models/2ph_comp/model.json`

"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import darts

# Make sure all modules are imported successfully


def valid_path(string):
    if os.path.exists(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"invalid path: '{string}'")


def get_lib_var():
    if sys.platform == 'linux':
        return 'LD_PRELOAD'
    elif sys.platform == 'darwin':
        return 'DYLD_LIBRARY_PATH'
    elif sys.platform.startswith('win'):
        return 'PATH'
    else:
        return None


def get_lib_search_var():
    if sys.platform == 'linux':
        return 'LD_LIBRARY_PATH'
    elif sys.platform == 'darwin':
        return 'DYLD_LIBRARY_PATH'
    elif sys.platform.startswith('win'):
        return 'PATH'
    else:
        return None


def get_darts_path():
    return Path(darts.__path__[0])


def main():
    args_list = sys.argv.copy()
    # Show help if no arguments are passed (same as 'darts -h')
    if len(args_list) <= 1:
        args_list.append('-h')

    # Handle multiprocessing spawn / resource_tracker callbacks
    if args_list[1] in ('-c', '-m'):
        # Extend dynamic loader search path for inline Python execution
        lib_search_var = get_lib_search_var()
        if lib_search_var:
            os.environ[lib_search_var] = (
                os.environ.get(lib_search_var, "") + os.pathsep + str(get_darts_path())
            )
        python_args = [sys.executable] + args_list[1:]
        res = subprocess.run(python_args)
        sys.exit(res.returncode)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a python script or JSON ModelSpec.",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const=True,
        help=(
            "If a path is provided and ends with .json, runs the JSON ModelSpec. "
            "If provided without a value and PATH is a folder, runs model.py in that folder."
        ),
        default=False,
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Simulation days when running JSON ModelSpec.",
    )
    parser.add_argument(
        "--report-days",
        type=float,
        default=None,
        help="Report days (alias of --report-days).",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        help="Set verbosity level: 0 for silent, 1 for normal, 2 for verbose.",
        default=1,
    )
    parser.add_argument(
        "--vtk-output",
        action="store_true",
        default=False,
        help="Output vtk files for each reporting step.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show program's version number and exit."
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments to pass to the script."
    )

    args = parser.parse_args(args_list[1:])

    def print_version():
        import pkg_resources

        version = pkg_resources.get_distribution("open-darts").version
        print(f"open-darts: v{version}")

    if args.version:
        print_version()
        exit()

    path = args.path
    python_args = [sys.executable]

    if not path and not isinstance(args.json, str):
        print_version()
        parser.print_usage()
        exit()

    # Arguments normalization
    report_days = args.report_days if args.report_days is not None else args.days

    # JSON ModelSpec via --model <file.json>
    model_json_path = None
    if isinstance(args.json, str) and args.json.lower().endswith('.json'):
        model_json_path = args.json

    # Helper: ensure runtime libs are set for child Python processes
    def _prepare_env_for_subprocess():
        # Prefer search path over forced preload to avoid ABI conflicts
        lib_search_var = get_lib_search_var()
        if lib_search_var:
            os.environ[lib_search_var] = (
                os.environ.get(lib_search_var, "") + os.pathsep + str(get_darts_path())
            )

        # Optional opt-in to force-preload libstdc++.so.6 if absolutely required
        if os.environ.get("DARTS_FORCE_PRELOAD_LIBSTDCXX", "0") in (
            "1",
            "true",
            "True",
        ):
            lib_var = get_lib_var()
            if lib_var:
                os.environ[lib_var] = str(get_darts_path() / "libstdc++.so.6") + (
                    ":" + os.environ.get(lib_var, "")
                    if os.environ.get(lib_var, "")
                    else ""
                )

    # JSON ModelSpec path handling via positional PATH
    if model_json_path or (
        path and isinstance(path, str) and path.lower().endswith(".json")
    ):
        # Run JSON-driven model via subprocess module to avoid in-process imports
        _prepare_env_for_subprocess()
        json_path = model_json_path if model_json_path else path
        run_args = [
            sys.executable,
            '-m',
            'darts.api.run_json_model',
            '--json',
            json_path,
        ]
        if args.days is not None:
            run_args += ['--days', str(args.days)]
        if report_days is not None:
            run_args += ['--report-days', str(report_days)]
        if args.vtk_output:
            run_args.append('--vtk-output')
        res = subprocess.run(run_args)
        sys.exit(res.returncode)

    python_args.append(path)
    python_args += args.args

    # Update env vars for running DARTS
    _prepare_env_for_subprocess()

    res = subprocess.run(python_args)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
