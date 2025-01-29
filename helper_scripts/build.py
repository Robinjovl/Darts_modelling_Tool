import os, sys, argparse


import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Script to install opendarts."
    )
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
        help="Enable Multi-thread MT (with OMP) build. Default: true",
    )
    parser.add_argument(
        "-G", action="store_true", help="Enable GPU build. Default: false"
    )
    parser.add_argument(
        "-r",
        action="store_true",
        help="Skip building thirdparty libraries. Default: false",
    )
    parser.add_argument(
        "-a",
        action="store_true",
        help="Update private artifacts bos_solvers for CI/CD. Default: false",
    )
    parser.add_argument("-b", type=str, metavar="SPATH", help="Path to bos_solvers.")
    parser.add_argument(
        "-d",
        type=str,
        metavar="MODE",
        default="Release",
        help="Configuration for C++ code [Release, Debug]. Default: Release",
    )
    parser.add_argument(
        "-j",
        type=int,
        metavar="N",
        default=8,
        help="Set number of threads (N) for compilation. Default: 8",
    )
    parser.add_argument(
        "-g", type=str, metavar="g++VER", help="Specify a compiler (g++) version."
    )

    args = parser.parse_args()

    clean_mode = args.c
    testing = args.t
    wheel = args.w
    MT = args.m
    GPU = args.G
    skip_req = args.r
    bos_solvers_artifact = args.a
    bos_solvers_dir = args.b
    config = args.d
    NT = args.j
    special_gpp = args.g is not None
    gpp_version = args.g if special_gpp else "g++"

    # Rest of the code follows here...

if __name__ == "__main__":
    main()
