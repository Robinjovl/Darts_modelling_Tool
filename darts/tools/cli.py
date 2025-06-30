"""
A CLI for DARTS, which ensures that the runtime environment is properly set up for
running DARTS scripts and models.

You can either manually specify the path to python scripts:
    `darts models/2ph_comp/main.py`

Or simply specify the path to a folder containg a main.py script:
    `darts models/2ph_comp`

Or directly run a model.py model script:
    `darts --model models/2ph_comp`
"""

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import darts

# Make sure all modules are imported successfully
from .. import discretizer, engines


def valid_path(string):
    if os.path.exists(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"invalid path: '{string}'")


def get_lib_var():
    if sys.platform == 'linux':
        return 'LD_LIBRARY_PATH'
    elif sys.platform == 'darwin':
        return 'DYLD_LIBRARY_PATH'
    elif sys.platform.startswith('win'):
        return 'PATH'
    else:
        return None


def get_darts_path():
    return Path(darts.__file__).parent


def main():
    # --- Handle multiprocessing spawn / resource_tracker callbacks ---
    # The spawn start method uses: python -c "...spawn_main(...)"
    # We detect '-c' or '-m' as the first passthrough argument and forward directly.
    if sys.argv[1] in ('-c', '-m'):
        lib_var = get_lib_var()
        if lib_var:
            # Prepend our DARTS library path to existing
            os.environ[lib_var] = (
                str(get_darts_path()) + os.pathsep + os.environ.get(lib_var, "")
            )
        # Forward directly to the real Python executable
        python_args = [sys.executable] + sys.argv[1:]
        res = subprocess.run(python_args)
        sys.exit(res.returncode)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "path",
        type=valid_path,
        nargs="?",
        help="Path to a python script or a folder containing a DARTS script.",
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="Optional boolean flag to indicate model usage.",
        default=False,
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
        "--version", action="store_true", help="Show program's version number and exit."
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments to pass to the script."
    )

    args = parser.parse_args()

    def print_version():
        import pkg_resources

        version = pkg_resources.get_distribution("open-darts").version
        print(f"open-darts: v{version}")

    if args.version:
        print_version()
        exit()
    path = args.path

    python_args = [sys.executable]

    if not path:
        print_version()
        parser.print_usage()
        exit()

    if path and os.path.isdir(path):
        file = "model.py" if args.model else "main.py"
        filepath = os.path.join(path, file)

        if os.path.isfile(filepath):
            path = filepath
        else:
            print(
                f"No '{file}' script found in '{path}'.\nPlease create one, or manually specify the file you want to run."
            )
            exit(1)

    if path:
        python_args.append(path)

    python_args += args.args

    # Update env vars for running DARTS
    lib_var = get_lib_var()

    if lib_var:
        os.environ[lib_var] = str(get_darts_path()) + ":" + os.environ.get(lib_var, "")

    res = subprocess.run(python_args)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
