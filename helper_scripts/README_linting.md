# Python Linting Setup

This document describes the Python linting setup for the DARTS project.

## Overview

The project uses three main linting tools to ensure code quality:

1. **flake8** - For syntax checking and style enforcement
2. **black** - For automatic code formatting
3. **isort** - For import sorting

## CI Pipeline Integration

The linting is integrated into the GitLab CI pipeline as a `pre_commit` stage. The linting job will run on:
- Merge requests
- Main branch commits
- Tagged releases
- When `RUN_LINTING=1` environment variable is set

## Local Development

### Prerequisites

Install the required linting tools:

```bash
pip install flake8 black isort
```

Or install the development dependencies:

```bash
pip install -e .[dev]
```

## Cross-Platform Linting Script

The project provides a single Python-based linting script that works on Linux, Windows, and macOS:

### Python Script (Cross-Platform)

The `lint_python.py` script works on all platforms:

```bash
# Check for linting issues (no fixes)
python helper_scripts/lint_python.py

# Check and automatically fix formatting issues
python helper_scripts/lint_python.py --fix

# Verbose output with fixes
python helper_scripts/lint_python.py --verbose --fix

# Check a specific directory
python helper_scripts/lint_python.py --dir path/to/darts
```

**Advantages:**
- Works on Linux, Windows, and macOS
- Better error handling and output formatting
- More maintainable code
- Consistent behavior across platforms
- No shell dependencies
- Used by both local development and CI pipeline

### Command Line Options

- `-f, --fix`: Automatically fix formatting issues (black and isort)
- `-v, --verbose`: Enable verbose output
- `-d, --dir DIR`: Specify darts directory (default: darts)
- `-h, --help`: Show help message

## Manual Tool Usage

You can also run the tools individually:

```bash
# Run flake8
flake8 darts/

# Check formatting with black
black --check --diff darts/

# Fix formatting with black
black darts/

# Check import sorting with isort
isort --check-only --diff darts/

# Fix import sorting with isort
isort darts/
```

## Configuration Files

### .flake8
Contains flake8 configuration with:
- Maximum line length: 88 characters
- Maximum complexity: 10
- Excluded directories and files
- Ignored error codes

### pyproject.toml
Contains configuration for:
- **black**: Line length 88, Python 3.10+ target
- **isort**: Black-compatible profile, known package categorization

## Pre-commit Hook (Optional)

You can set up a pre-commit hook to automatically run linting before each commit:

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

3. Install the hook:
```bash
pre-commit install
```

## Platform-Specific Notes

### All Platforms
- Use `lint_python.py` - the single cross-platform solution
- Ensure Python is in your PATH
- The script works identically across all operating systems

### Cross-Platform Development
- The Python script (`lint_python.py`) is used by both local development and CI pipeline
- Configuration files (`.flake8`, `pyproject.toml`) work identically across platforms
- No platform-specific setup required

## Troubleshooting

### Common Issues

1. **Line length errors**: The project uses 88 characters as the maximum line length (compatible with black)
2. **Import sorting issues**: Use `isort` to automatically sort imports
3. **Formatting issues**: Use `black` to automatically format code
4. **Python not found**: Ensure Python is installed and in your PATH

### Getting Help

- Run `python helper_scripts/lint_python.py --help` for usage information
- Check the tool documentation:
  - [flake8](https://flake8.pycqa.org/)
  - [black](https://black.readthedocs.io/)
  - [isort](https://pycqa.github.io/isort/) 