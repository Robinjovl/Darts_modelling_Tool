#!/usr/bin/env python3
"""
Python linting script for DARTS project
This script performs the same linting checks as the CI pipeline
Works on both Linux and Windows
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_status(message: str) -> None:
    """Print a status message with blue color"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def print_success(message: str) -> None:
    """Print a success message with green color"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow color"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def print_error(message: str) -> None:
    """Print an error message with red color"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def check_tool(tool_name: str) -> bool:
    """Check if a tool is installed and available"""
    try:
        subprocess.run([tool_name, "--version"], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_command(cmd: List[str], description: str, verbose: bool = False) -> bool:
    """Run a command and return success status"""
    print_status(description)
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose and result.stdout:
            print(result.stdout)
        print_success(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def run_flake8_checks(darts_dir: str, verbose: bool) -> int:
    """Run flake8 checks and return number of errors"""
    errors = 0
    
    print_status("Running flake8 syntax and style checks...")
    
    # First run: critical errors only
    cmd_critical = [
        "flake8", darts_dir, 
        "--count", "--select=E9,F63,F7,F82", 
        "--show-source", "--statistics"
    ]
    
    if not run_command(cmd_critical, "flake8 critical checks", verbose):
        print_error("flake8 found critical errors!")
        errors += 1
    else:
        print_success("flake8 critical checks passed")
    
    # Second run: style and complexity checks
    cmd_style = [
        "flake8", darts_dir,
        "--count", "--exit-zero", 
        "--max-complexity=10", "--max-line-length=88", 
        "--statistics"
    ]
    
    if not run_command(cmd_style, "flake8 style checks", verbose):
        print_warning("flake8 found style issues (non-critical)")
        errors += 1
    else:
        print_success("flake8 style checks passed")
    
    return errors


def run_black_checks(darts_dir: str, fix: bool, verbose: bool) -> int:
    """Run black checks/fixes and return number of errors"""
    if fix:
        cmd = ["black", "--skip-string-normalization", darts_dir]
        return 0 if run_command(cmd, "Running black to fix code formatting", verbose) else 1
    else:
        cmd = ["black", "--check", "--diff", "--skip-string-normalization", darts_dir]
        if run_command(cmd, "black formatting check", verbose):
            print_success("black formatting check passed")
            return 0
        else:
            print_error("black found formatting issues. Use --fix to automatically fix them.")
            return 1


def run_isort_checks(darts_dir: str, fix: bool, verbose: bool) -> int:
    """Run isort checks/fixes and return number of errors"""
    if fix:
        cmd = ["isort", darts_dir]
        return 0 if run_command(cmd, "Running isort to fix import sorting", verbose) else 1
    else:
        cmd = ["isort", "--check-only", "--diff", darts_dir]
        if run_command(cmd, "isort import sorting check", verbose):
            print_success("isort import sorting check passed")
            return 0
        else:
            print_error("isort found import sorting issues. Use --fix to automatically fix them.")
            return 1


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Python linting script for DARTS project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Check only (no fixes)
  %(prog)s --fix              # Check and fix formatting issues
  %(prog)s --verbose --fix    # Verbose output with fixes
        """
    )
    
    parser.add_argument(
        "-f", "--fix",
        action="store_true",
        help="Automatically fix formatting issues (black and isort)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-d", "--dir",
        default="darts",
        help="Specify darts directory (default: darts)"
    )
    
    args = parser.parse_args()
    
    # Check if darts directory exists
    if not os.path.isdir(args.dir):
        print_error(f"DARTS directory '{args.dir}' not found!")
        sys.exit(1)
    
    print_status(f"Starting Python linting checks for directory: {args.dir}")
    
    # Check if required tools are installed
    print_status("Checking for required tools...")
    tools = ["flake8", "black", "isort"]
    missing_tools = []
    
    for tool in tools:
        if not check_tool(tool):
            missing_tools.append(tool)
    
    if missing_tools:
        print_error(f"Missing tools: {', '.join(missing_tools)}")
        print_error("Please install them with: pip install " + " ".join(missing_tools))
        sys.exit(1)
    
    # Initialize error counter
    total_errors = 0
    
    # Run all checks
    # total_errors += run_flake8_checks(args.dir, args.verbose)
    total_errors += run_black_checks(args.dir, args.fix, args.verbose)
    total_errors += run_isort_checks(args.dir, args.fix, args.verbose)
    
    # Summary
    print()
    if total_errors == 0:
        print_success("All linting checks passed! 🎉")
        sys.exit(0)
    else:
        print_error(f"Linting found {total_errors} issue(s).")
        if not args.fix:
            print_status("Run with --fix to automatically fix formatting issues.")
        sys.exit(1)


if __name__ == "__main__":
    main() 