from os.path import isfile
from pathlib import Path
import argparse, os, subprocess, sys


def valid_path(string):
    if os.path.exists(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"invalid path: '{string}'")


def main():
    # Make sure all modules are imported successfully
    from .. import engines, discretizer

    parser = argparse.ArgumentParser(description="CLI tool for running DARTS models.")

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

    if not path:
        print_version()
        print("Please supply the path to a DARTS script.")
        exit()

    if os.path.isdir(path):
        file = "model.py" if args.model else "main.py"
        filepath = os.path.join(path, file)

        if os.path.isfile(filepath):
            path = filepath
        else:
            print(
                f"No '{file}' script found in '{path}'.\nPlease create one, or manually specify the file you want to run."
            )
            exit(1)
    # Define the new environment variable
    new_env = {"LD_LIBRARY_PATH": str(Path(__file__).parent.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}

    # Update the environment of the current process
    env = {**os.environ, **new_env}

    res = subprocess.run([sys.executable, path] + args.args, env=env)
    sys.exit(res.returncode)
