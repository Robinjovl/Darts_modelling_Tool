### Agent quick commands and conventions

- **Use absolute paths**
- **Run non-interactively** and print the command before execution.
- **Project-level commands** should be executed from the repo root.

### Lint Python files

```bash
pre-commit run -v --files <absolute-path-to-file> --show-diff-on-failure
```

### Run a model

```bash
darts <absolute-path-to-model-folder>/main.py
```

### Build/install Python package locally (from repo root)

```bash
./helper_scripts/install_darts.sh -e
```

### Git rules

- Baseline branch is `development`
- Keep commits focused and include a brief but descriptive message.
