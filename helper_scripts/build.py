import os, argparse, shutil
import platform
from pathlib import Path
import subprocess
from argparse import ArgumentTypeError
from glob import glob
from sys import flags


def valid_dir(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"invalid dir: '{string}'")


def validator(obj, condition, name, err_msg=""):
    if not condition():
        if err_msg:
            err_msg = "\n" + err_msg
        raise argparse.ArgumentTypeError(f"invalid {name}: '{obj}'{err_msg}")

    return obj


RESET_COLOR = "\033[0m"  # ANSI escape code to reset color


def warning(*args, sep=" ", **kwargs):
    WARNING_COLOR = "\033[93m"  # ANSI escape code for yellow
    print(f"{WARNING_COLOR}Warning: {sep.join(args)}{RESET_COLOR}", **kwargs)


def error(*args, sep=" ", **kwargs):
    ERROR_COLOR = "\033[91m"  # ANSI escape code for red
    print(f"{ERROR_COLOR}Error: {sep.join(args)}{RESET_COLOR}", **kwargs)
    exit(1)


def positive_int(n: str) -> int:
    if not n.isdigit():
        raise ArgumentTypeError(f"invalid positive int: '{n}'")

    return int(n)


parser = argparse.ArgumentParser(description="Script to install opendarts.")
parser.add_argument(
    "-c",
    action="store_true",
    help="Cleans up build to prepare a new fresh build. Default: don't clean",
)
parser.add_argument(
    "-t",
    action="store_true",
    help="Enable testing: ctest of solvers. Default: don't test",
)
parser.add_argument(
    "-w",
    action="store_true",
    help="Enable generation of python wheel. Default: false",
)
parser.add_argument(
    "-m",
    action="store_true",
    default=True,
    help="Enable Multi-thread MT (with OMP) build. Default: true",
)
parser.add_argument(
    "-s", "--silent", action="store_true", help="Silence most build commands"
)
parser.add_argument("-G", action="store_true", help="Enable GPU build. Default: false")
parser.add_argument(
    "-r",
    action="store_true",
    help="Skip building thirdparty libraries. Default: false",
)
parser.add_argument(
    "-a",
    action="store_true",
    help="Fetch and update private artifacts bos_solvers for CI/CD. Default: false",
)
parser.add_argument("-b", type=valid_dir, metavar="SPATH", help="Path to bos_solvers.")

builds_configs = ["Release", "Debug"]


def valid_build_config(config: str):
    if config in builds_configs:
        return config
    else:
        raise argparse.ArgumentTypeError(
            f"invalid build config: '{config}'\nOptions: {builds_configs}"
        )


parser.add_argument(
    "-d",
    type=valid_build_config,
    metavar="MODE",
    default="Release",
    help="Configuration for C++ code [Release, Debug]. Default: Release",
)
parser.add_argument(
    "-j",
    type=positive_int,
    metavar="N",
    default=8,
    help="Set number of threads (N) for compilation. Default: 8",
)
parser.add_argument(
    "-g", type=str, metavar="g++VER", help="Specify a compiler (g++) version."
)

args = parser.parse_args()

clean = args.c
testing = args.t
wheel = args.w
mt = args.m
gpu = args.G
skip_req = args.r
bos_solvers_artifact = args.a
bos_solvers_dir = args.b
config = args.d
num_threads = args.j
special_gpp = args.g is not None
gpp_version = args.g if special_gpp else "g++"

# Rest of the code follows here...

iter_solvers = bos_solvers_artifact or bos_solvers_dir

# Amend possible contradictory inputs

if iter_solvers and testing:
    # tests are only available in open-DARTS, bos_solvers
    # do not have testing
    warning("BOS solvers do not have testing, disabling testing.")
    testing = False

if not iter_solvers:
    if gpu:
        error("GPU build requires GPU BOS solvers. Specify the path with -b.")
    elif mt:
        warning(
            "Open-DARTS linear solvers do not support multi-threading. Switched to the sequentional build."
        )
        mt = False


# ---------------------------------------------------------------------


# Get path to root dir
base_dir = Path(os.path.dirname(__file__)).parent
os.chdir(base_dir)

if clean:
    print("\nCleaning build folders, dist folder and generated python modules.")
    thirdpary_dirs = ["eigen", "pybind11", "mshIO", "hypre"]
    thirdpary_dirs = [os.path.join("thirdparty", "build", d) for d in thirdpary_dirs]

    build_dirs = ["dist", "build"]

    for dir in build_dirs + thirdpary_dirs:
        shutil.rmtree(dir, True)

    for file in glob("darts/*.so"):
        os.remove(file)


# Build ---------------------------------------------------------------
def title(message: str):
    print(f"\n- {message}\n")


def subtitle(message: str):
    print(f"\n-- {message}\n")


def run(command, log_file=None, errs_to_screen=True, stdout_only_screen=False):
    if isinstance(command, str):
        command = command.split()
    process = subprocess.Popen(
        command, stdout=None if stdout_only_screen else log_file, stderr=subprocess.PIPE, text=True
    )
    while True:
        output = process.stderr.readline()

        if output == "" and process.poll() is not None:
            break

        if output:
            if errs_to_screen:
                print(output, end="")
            log_file.write(output)


def cmake(source: str, build: str, flags: list[str] = [], log_file=None):

    command = [
        "cmake",
        "-S",
        source,
        "-B",
        build,
        "-D",
        f"CMAKE_INSTALL_PREFIX={os.path.join(thirdparty, "install")}",
        "-Wno-dev",
    ]

    for flag in flags:
        command.append("-D")
        command.append(flag)

    run(
        command,
        log_file,
    )


def make_install(path=None, log_file=None, errs_to_screen=True, stdout_only_screen=False):
    make_command = ["make", "install", "-s", "-j", str(num_threads)]
    if path:
        make_command.append("-C")
        make_command.append(path)
    run(make_command, log_file, errs_to_screen, stdout_only_screen)


def open_log(name: str):
    return open(os.path.join(base_dir, f"make_{name}.log"), "w")


def build(name: str, source: str, build: str, flags: list[str] = []):
    subtitle(f"Install {name}")

    short_name = name.split()[0].lower()
    with open_log(short_name) as log_file:
        print("    Generating build files...", end="", flush=True)
        # if flags:
        #     print(" with flags: ", *flags)
        cmake(source, build, flags, log_file)
        print(" DONE.")
        print("    Compiling...", end="", flush=True)
        make_install(build, log_file)
        print(" DONE.")


thirdparty = os.path.join(base_dir, "thirdparty")

if (
    skip_req
    and os.path.exists(os.path.join(thirdparty, "install/lib/libHYPRE.a"))
    and os.path.exists(os.path.join(thirdparty, "install/include/eigen3"))
    and os.path.exists(os.path.join(thirdparty, "SuperLU_5.2.1/SRC/libsuperlu_5.1.a"))
):
    title("Requirements already installed")
else:
    title("Update submodules: START")

    subprocess.run(["git", "submodule", "sync", "--recursive"])
    subprocess.run(["git", "submodule", "update", "--recursive", "--init"])

    title("Update submodules: DONE!")

    # Install requirements
    title("Install requirements: START")

    build(
        "EIGEN 3",
        os.path.join(thirdparty, "eigen"),
        os.path.join(thirdparty, "build", "eigen"),
    )

    # Setup hypre build with no MPI support (we only use single processor)
    # Request build of tests and examples just to be sure everything is fine in the build
    # For debugging: -DHYPRE_ENABLE_PRINT
    build(
        "Hypre",
        source=os.path.join(thirdparty, "hypre", "src"),
        build=os.path.join(thirdparty, "hypre", "src", "cmbuild"),
        flags=[
            "HYPRE_BUILD_TESTS=ON",
            "HYPRE_BUILD_EXAMPLES=ON",
            "HYPRE_WITH_MPI=OFF",
        ],
    )

    subtitle("Install SuperLU")
    os.chdir(thirdparty)
    os.chdir("SuperLU_5.2.1")

    if platform.system() == "Darwin":
        shutil.copy("conf_gcc-11_macOS_m1.mk", "conf.mk")
        shutil.copy("make_gcc-11_macOS_m1.inc", "make.inc")
    else:
        shutil.copy("conf_gcc_linux.mk", "conf.mk")
        shutil.copy("make_gcc_linux.inc", "make.inc")

    with open_log("superlu") as log_file:
        print("    Compiling...", end="", flush=True)
        run(f"make -j {num_threads}", log_file=log_file)
        make_install(log_file=log_file)
        print(" Done.")

    if bos_solvers_artifact:
        print("TODO: update private artifacts")

    title("Install requirements: DONE! ")

if bos_solvers_artifact:
    bos_solvers_dir = os.path.join(base_dir, "engines/lib/darts_linear_solvers")

line_width = 72
    
def big_title(message: str, char="="):
    print("\n" + line_width * char)
    print(f"| {message}")
    print(line_width * char + "\n")

big_title("Building openDARTS: START")

cmake_options = [f"CMAKE_BUILD_TYPE={config}"]

if testing:
    cmake_options.append("ENABLE_TESTING=ON")

if special_gpp:
    cmake_options.append(f"CMAKE_CXX_COMPILER={gpp_version}")

build_type = "ST"

if gpu:
    build_type = "GPU"
elif mt:
    build_type = "MT"

cmake_options.append(f"OPENDARTS_CONFIG={build_type}")


if bos_solvers_dir:
    cmake_options.append(f"BOS_SOLVERS_DIR={bos_solvers_dir}")

print("CMake options:", *cmake_options, end="\n\n")

build_dir = os.path.join(base_dir, "build")
build("darts", base_dir, build_dir, flags=cmake_options)

# Test
if testing:
    run(["ctest", "--test-dir", os.path.join(thirdparty, "build")])

big_title("Building openDARTS: DONE!")

def py_title(message: str):
    big_title(message, char="*")

py_title("Installing python package open-darts: START")

# generating build info of darts-package
subprocess.run(["python3", "darts/print_build_info.py"], check=True)

# build darts.whl
if wheel:
    subtitle("Generating python wheel...")
    subprocess.run(["python3", "setup.py", "clean"], check=True)
    with open_log("wheel") as log_file:
        subprocess.run(["python3", "setup.py", "build", "bdist_wheel"], stdout=log_file, stderr=subprocess.STDOUT, check=True)
    subtitle("Python wheel generated!\n")

# installing python package
subprocess.run(["python3", "-m", "pip", "install", "-e", "."], check=True)

py_title("Installing python package open-darts: DONE!")
