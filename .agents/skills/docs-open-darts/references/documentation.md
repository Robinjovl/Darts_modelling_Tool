# Skill: Building and Maintaining Documentation

## Overview

open-DARTS documentation is built with **Sphinx** using the **Read the Docs**
theme. Source files use both reStructuredText (`.rst`) and Markdown (`.md` via
`myst_parser`). API documentation is auto-generated from Python docstrings.

---

## Documentation Structure

```
docs/
├── conf.py               # Sphinx configuration
├── Makefile              # Build driver (Linux/macOS)
├── make.bat              # Build driver (Windows)
├── index.rst             # Root document
├── api.rst               # Auto-generated API reference
├── references.bib        # Bibliography (BibTeX)
├── about_darts/          # About open-DARTS
├── getting_started/      # Installation, tutorial, FAQ
│   ├── installation.md
│   ├── tutorial.md
│   ├── example_models.md
│   ├── supported_features.md
│   ├── troubleshooting.md
│   ├── F.A.Q..md
│   └── configure_hardware.md
├── for_developers/       # Developer guides
│   ├── darts_gitlab_setup.md
│   └── linting.md
└── technical_reference/  # Technical docs
    ├── glossary.md
    ├── reservoir.md
    ├── wells.md
    └── units.md
```

---

## Sphinx Configuration (docs/conf.py)

| Setting | Value |
|---|---|
| Theme | `sphinx_rtd_theme` |
| Source formats | `.rst`, `.md` |
| Auto-doc | `sphinx.ext.autodoc`, `sphinx.ext.napoleon` |
| Summaries | `sphinx.ext.autosummary` |
| Source links | `sphinx.ext.viewcode` |
| Copy button | `sphinx_copybutton` |
| Bibliography | `sphinxcontrib.bibtex` |
| Markdown support | `myst_parser` with `linkify` extension |

---

## Building Documentation Locally

### Install Documentation Dependencies

```bash
pip install .[docs]
# or individually:
pip install myst-parser sphinx_rtd_theme sphinx sphinx-tabs sphinx_inline_tabs \
    sphinxcontrib-matlabdomain sphinxcontrib-bibtex linkify-it-py sphinx-copybutton
```

### Build HTML Documentation

```bash
cd docs
make html
# Output: docs/_build/html/index.html
```

Or using sphinx-build directly:

```bash
cd docs
sphinx-build -b html . _build/html
```

### Other Build Targets

```bash
make help          # List available targets
make latexpdf      # PDF via LaTeX
make linkcheck     # Verify external links
make doctest       # Run doctest blocks
make clean         # Remove build artifacts
```

---

## API Documentation

API docs are generated from Python docstrings using `autodoc` and `napoleon`.
The entry point is `docs/api.rst`.

### Docstring Style

Put opening and closing triple quotation marks on separate lines. Document
input and output arguments with `:param`, `:type`, `:return:`, and `:rtype:`
fields. Example:

```python
def compute_density(pressure, temperature):
    """
    Compute fluid density at given conditions.

    :param pressure: Pressure [bar]
    :type pressure: float
    :param temperature: Temperature [K]
    :type temperature: float
    :return: Fluid density [kg/m3]
    :rtype: float
    """
```

### Regenerating API Docs

If new modules are added, update `docs/api.rst` to include them in the
`autosummary` directives.

---

## CI Documentation Deployment

Documentation is built and deployed in the CI pipeline
(`.cicd/jobs/deploy.yml`, `pages` job):

1. Install `open-darts` with docs dependencies: `pip install .[docs]`
2. Install the built wheel
3. Build: `sphinx-build -b html . public`
4. Deploy to GitLab Pages

The job triggers on:
- Pushes to `main`
- Version tags (`v#.#.#`)
- Manual trigger with `DOCS_PAGES=1`

Published at: <https://open-darts.readthedocs.io/en/docs>

---

## Adding New Documentation

### New Markdown Page

1. Create a `.md` file in the appropriate subdirectory under `docs/`.
2. Add it to the `toctree` in the parent `.rst` or `.md` file.
3. Use MyST Markdown syntax (superset of CommonMark).

### New API Module

1. Add the module path to `docs/api.rst` in an `autosummary` block.
2. Ensure the module has proper docstrings.
3. Rebuild to verify.

### Bibliography

Add references to `docs/references.bib` in BibTeX format. Cite in docs with:
- RST: `` :cite:`key` ``
- Markdown (MyST): `` {cite}`key` ``

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `myst_parser` not found | `pip install .[docs]` |
| Import errors during autodoc | Ensure `open-darts` is installed in the same env |
| Missing module in API docs | Add to `autosummary` in `api.rst` |
| Broken cross-references | Check label syntax; use `{ref}` in MyST |
| Bibliography not rendering | Verify `references.bib` path in `conf.py` |
