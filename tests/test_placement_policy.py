"""Unit tests for the ``darts/`` placement-policy checker (M4).

The checker under test is ``helper_scripts/check_placement_policy.py``: it
enforces the two mechanically decidable clauses of the placement policy
(MR !305 review, section 6) -- every module in the installable package must have
an in-repo consumer, and no module in the package may import from ``models/``.

Each test builds a throwaway repository in ``tmp_path`` with the same shape as
the real one (``darts/``, ``models/``, ``tests/``) and runs the checker against
it, so the fixtures exercise the production configuration rather than a
test-only one.

The interesting cases are the false-positive ones. A module can be genuinely
used without anyone importing it by name -- this codebase selects drift-flux
closures with ``drift_flux_model="tang_2019"`` -- and a module can look used
while being dead, because one line in a package ``__init__`` re-exports it.
Both directions are pinned here, including the negative twin of the registry
case, so that the registry handling cannot decay into a blanket exemption.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "helper_scripts" / "check_placement_policy.py"

if not CHECKER_PATH.exists():  # pragma: no cover - wheel-only checkouts
    pytest.skip(
        "helper_scripts/check_placement_policy.py is not in this checkout",
        allow_module_level=True,
    )


def _load_checker():
    name = "_check_placement_policy"
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def write_tree(root: Path, files: dict) -> Path:
    """Materialise ``{relative path: source}`` under ``root``."""
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def write_allowlist(root: Path, entries: list) -> Path:
    path = root / "allowlist.json"
    path.write_text(json.dumps({"allow": entries}), encoding="utf-8")
    return path


def run(root: Path, allowlist: Path = None):
    """Run both clauses; return the list of unsuppressed violations."""
    path = allowlist if allowlist is not None else root / "no-allowlist.json"
    violations, _stale, _used = checker.check(root, path)
    return violations


def flagged(violations, clause=None):
    return sorted(v.module for v in violations if clause is None or v.clause == clause)


#: A minimal package skeleton every fixture starts from.
BASE = {
    "darts/__init__.py": "",
    "darts/pkg/__init__.py": "",
}


# --------------------------------------------------------------------------- #
# Clause 1 -- zero-consumer
# --------------------------------------------------------------------------- #


def test_module_with_no_importer_is_flagged(tmp_path):
    """The choke case: a module nothing in the repository imports."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/lonely.py": "class Choke:\n    pass\n",
            "darts/pkg/used.py": "class Pipe:\n    pass\n",
            "models/case/model.py": "from darts.pkg.used import Pipe\n",
        },
    )
    violations = run(tmp_path)
    assert flagged(violations) == ["darts.pkg.lonely"]
    assert violations[0].clause == "zero-consumer"
    assert "darts/pkg/lonely.py" in violations[0].render()
    assert "no importer" in violations[0].render()


def test_module_imported_only_by_a_test_passes(tmp_path):
    """A unit test is a legitimate consumer -- it is what CI exercises."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/viscosity.py": "class AirViscosity:\n    pass\n",
            "tests/test_viscosity.py": (
                "from darts.pkg.viscosity import AirViscosity\n\n\n"
                "def test_it():\n    assert AirViscosity\n"
            ),
        },
    )
    assert run(tmp_path) == []


def test_module_imported_only_by_a_model_passes(tmp_path):
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/plotter.py": "def plot():\n    pass\n",
            "models/case/main.py": "from darts.pkg.plotter import plot\n\nplot()\n",
        },
    )
    assert run(tmp_path) == []


def test_import_seen_through_alias_and_conditional_import(tmp_path):
    """Aliased imports nested inside try/except and functions still count."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/optional.py": "VALUE = 1\n",
            "models/case/main.py": (
                "def load():\n"
                "    try:\n"
                "        import darts.pkg.optional as opt\n"
                "    except ImportError:\n"
                "        opt = None\n"
                "    return opt\n"
            ),
        },
    )
    assert run(tmp_path) == []


def test_mentions_in_comments_and_prose_do_not_count(tmp_path):
    """The scan is AST-based: a commented-out import is not a consumer."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/lonely.py": "class Choke:\n    pass\n",
            "models/case/main.py": (
                '"""Post-processing. See darts.pkg.lonely for the choke model."""\n'
                "\n"
                "# from darts.pkg.lonely import Choke  # disabled for now\n"
            ),
        },
    )
    assert flagged(run(tmp_path)) == ["darts.pkg.lonely"]


def test_dynamic_import_by_string_counts(tmp_path):
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/dyn.py": "VALUE = 1\n",
            "models/case/main.py": (
                "import importlib\n\nmod = importlib.import_module('darts.pkg.dyn')\n"
            ),
        },
    )
    assert run(tmp_path) == []


def test_package_init_is_not_itself_flagged(tmp_path):
    """A package exists so its modules are importable; it is not judged alone."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/used.py": "VALUE = 1\n",
            "models/case/main.py": "from darts.pkg.used import VALUE\n",
        },
    )
    assert run(tmp_path) == []


# --------------------------------------------------------------------------- #
# False positive class 1 -- package __init__ re-exports
# --------------------------------------------------------------------------- #


def test_reexport_alone_does_not_rescue_a_dead_module(tmp_path):
    """One line in __init__.py must not exempt a module nothing else uses."""
    write_tree(
        tmp_path,
        {
            "darts/__init__.py": "",
            "darts/pkg/__init__.py": "from darts.pkg.dead import Dead\n",
            "darts/pkg/dead.py": "class Dead:\n    pass\n",
            "models/case/main.py": "import darts\n",
        },
    )
    assert flagged(run(tmp_path)) == ["darts.pkg.dead"]


def test_reexported_symbol_used_by_a_consumer_counts(tmp_path):
    """``from darts.pkg import Thing`` consumes the module defining ``Thing``."""
    write_tree(
        tmp_path,
        {
            "darts/__init__.py": "",
            "darts/pkg/__init__.py": (
                "from darts.pkg.solver import Solver\n\n__all__ = ['Solver']\n"
            ),
            "darts/pkg/solver.py": "class Solver:\n    pass\n",
            "models/case/main.py": "from darts.pkg import Solver\n\nSolver()\n",
        },
    )
    assert run(tmp_path) == []


def test_reexported_symbol_used_as_an_attribute_counts(tmp_path):
    """``import darts.pkg`` then ``darts.pkg.Solver`` is consumption too."""
    write_tree(
        tmp_path,
        {
            "darts/__init__.py": "",
            "darts/pkg/__init__.py": "from darts.pkg.solver import Solver\n",
            "darts/pkg/solver.py": "class Solver:\n    pass\n",
            "models/case/main.py": "import darts.pkg\n\ndarts.pkg.Solver()\n",
        },
    )
    assert run(tmp_path) == []


# --------------------------------------------------------------------------- #
# False positive class 2 -- string-keyed factory registries
# --------------------------------------------------------------------------- #

#: A closure package shaped like ``darts/pipes`` after the closures are split
#: into one file each: the package ``__init__`` is the registry, and the models
#: only ever name the closure with a string.
REGISTRY_TREE = {
    "darts/__init__.py": "",
    "darts/pipes/__init__.py": "",
    "darts/pipes/closures/__init__.py": (
        "from darts.pipes.closures.tang import Tang2019Closure\n"
        "from darts.pipes.closures.shi import ShiT2WellClosure\n"
        "\n"
        "DRIFT_FLUX_CLOSURES = {\n"
        "    'shi_t2well': ShiT2WellClosure,\n"
        "    'tang_2019': Tang2019Closure,\n"
        "}\n"
    ),
    "darts/pipes/closures/tang.py": (
        "class Tang2019Closure:\n    name = 'tang_2019'\n"
    ),
    "darts/pipes/closures/shi.py": (
        "class ShiT2WellClosure:\n    name = 'shi_t2well'\n"
    ),
    "darts/pipes/pipe.py": (
        "from darts.pipes.closures import DRIFT_FLUX_CLOSURES\n"
        "\n"
        "class Pipe:\n"
        "    def __init__(self, drift_flux_model='shi_t2well'):\n"
        "        self.closure = DRIFT_FLUX_CLOSURES[drift_flux_model]()\n"
    ),
}


def test_registry_selected_module_is_not_flagged(tmp_path):
    """A closure chosen only by name is consumed, though nothing imports it."""
    write_tree(
        tmp_path,
        {
            **REGISTRY_TREE,
            "models/case/model.py": (
                "from darts.pipes.pipe import Pipe\n"
                "\n"
                "pipe = Pipe(drift_flux_model='tang_2019')\n"
            ),
            "tests/test_shi.py": (
                "from darts.pipes.pipe import Pipe\n"
                "\n"
                "def test_default():\n"
                "    assert Pipe(drift_flux_model='shi_t2well')\n"
            ),
        },
    )
    assert run(tmp_path) == []


def test_registry_key_no_consumer_names_is_still_flagged(tmp_path):
    """The negative twin: registry membership alone is not an exemption.

    ``shi_t2well`` is selected by a model, ``tang_2019`` is not selected
    anywhere -- so the Tang closure is dead code even though it sits in the
    same registry dictionary as the live one.
    """
    write_tree(
        tmp_path,
        {
            **REGISTRY_TREE,
            "models/case/model.py": (
                "from darts.pipes.pipe import Pipe\n"
                "\n"
                "pipe = Pipe(drift_flux_model='shi_t2well')\n"
            ),
        },
    )
    assert flagged(run(tmp_path)) == ["darts.pipes.closures.tang"]


def test_registry_registered_by_call_counts(tmp_path):
    """``register("name", Cls)`` is recognised as well as a dict literal.

    ``wiring.py`` is a side-effect registration module: the package imports it
    for its ``register()`` call and nothing else ever names it, so only the key
    it installs can show that it is exercised.
    """
    write_tree(
        tmp_path,
        {
            "darts/__init__.py": "",
            "darts/pkg/__init__.py": "import darts.pkg.wiring  # noqa: F401\n",
            "darts/pkg/registry.py": (
                "REGISTRY = {}\n\ndef register(key, obj):\n    REGISTRY[key] = obj\n"
            ),
            "darts/pkg/hooks.py": "class Hook:\n    pass\n",
            "darts/pkg/wiring.py": (
                "from darts.pkg.hooks import Hook\n"
                "from darts.pkg.registry import register\n"
                "\n"
                "register('lateral_heat', Hook)\n"
            ),
            "models/case/main.py": (
                "from darts.pkg.registry import REGISTRY\n\nREGISTRY['lateral_heat']\n"
            ),
        },
    )
    # hooks.py is consumed by the literal import in wiring.py; wiring.py itself
    # is consumed only through the model's use of the key it installs.
    assert flagged(run(tmp_path)) == []


def test_a_data_dictionary_is_not_a_registry(tmp_path):
    """Ordinary dict keys must not vouch for the module that builds them.

    ``darts/tools/jacobian.py`` builds ``{"rows": ..., "rhs": ...}`` inside a
    function; without both guards -- module level only, and selector-shaped
    keys only -- every model that mentions ``"rhs"`` would mark it consumed.
    """
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/dump.py": (
                "def save(rows, rhs):\n    return {'rows': rows, 'rhs': rhs}\n"
            ),
            "models/case/main.py": (
                "data = {'rows': [], 'rhs': []}\n\nprint(data['rhs'])\n"
            ),
        },
    )
    assert flagged(run(tmp_path)) == ["darts.pkg.dump"]


def test_module_level_registry_with_a_generic_key_is_not_trusted(tmp_path):
    """Even at module level, ``{"rhs": Cls}`` is too generic to be evidence."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/thing.py": "class Thing:\n    pass\n",
            "darts/pkg/table.py": (
                "from darts.pkg.thing import Thing\n\nTABLE = {'rhs': Thing}\n"
            ),
            "models/case/main.py": "print('rhs')\n",
        },
    )
    # table.py has no importer, and 'rhs' in a model does not make it one.
    assert flagged(run(tmp_path)) == ["darts.pkg.table"]


# --------------------------------------------------------------------------- #
# Clause 2 -- layering
# --------------------------------------------------------------------------- #


def test_darts_module_importing_models_is_flagged(tmp_path):
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/leaky.py": "from models.case.model import Model\n",
            "models/case/__init__.py": "",
            "models/case/model.py": "class Model:\n    pass\n",
            "tests/test_leaky.py": "from darts.pkg.leaky import Model\n",
        },
    )
    violations = run(tmp_path)
    assert flagged(violations, clause="layering") == ["darts.pkg.leaky"]
    rendered = violations[0].render()
    assert "outside the installable package" in rendered
    assert "darts/pkg/leaky.py" in rendered


def test_darts_module_importing_a_bare_model_name_is_flagged(tmp_path):
    """``import model`` is the form that actually occurs under ``models/``."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/leaky.py": "import model\n",
            "models/case/model.py": "class Model:\n    pass\n",
            "tests/test_leaky.py": "import darts.pkg.leaky\n",
        },
    )
    assert flagged(run(tmp_path), clause="layering") == ["darts.pkg.leaky"]


def test_sibling_relative_import_is_not_a_layering_violation(tmp_path):
    """A bare sibling import inside the package is a different defect.

    ``darts/reservoirs/mesh/geometry/main.py`` does ``from shapes import *``
    with ``shapes`` its own sibling; a model file of the same name must not
    turn that into a false layering report.
    """
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/shapes.py": "class Circle:\n    pass\n",
            "darts/pkg/geom.py": "from shapes import Circle\n",
            "models/case/shapes.py": "class Other:\n    pass\n",
            "tests/test_geom.py": ("import darts.pkg.geom\nimport darts.pkg.shapes\n"),
        },
    )
    assert flagged(run(tmp_path), clause="layering") == []


def test_layering_clean_tree_passes(tmp_path):
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/good.py": "import numpy\n",
            "tests/test_good.py": "import darts.pkg.good\n",
        },
    )
    assert flagged(run(tmp_path), clause="layering") == []


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #

GOOD_JUSTIFICATION = (
    "Deliberate library seed: general, documented, and kept in the package "
    "while its first CI-wired consumer is written."
)


def test_allowlist_suppresses_a_named_violation(tmp_path):
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/seed.py": "class Seed:\n    pass\n",
            "models/case/main.py": "import darts\n",
        },
    )
    allowlist = write_allowlist(
        tmp_path,
        [
            {
                "module": "darts.pkg.seed",
                "clause": "zero-consumer",
                "justification": GOOD_JUSTIFICATION,
            }
        ],
    )
    violations, stale, used = checker.check(tmp_path, allowlist)
    assert violations == []
    assert stale == []
    assert [entry.module for entry in used] == ["darts.pkg.seed"]


def test_allowlist_is_clause_specific(tmp_path):
    """Exempting a module from one clause does not exempt it from the other."""
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/leaky.py": "from models.case.model import Model\n",
            "models/case/__init__.py": "",
            "models/case/model.py": "class Model:\n    pass\n",
            "tests/test_leaky.py": "from darts.pkg.leaky import Model\n",
        },
    )
    allowlist = write_allowlist(
        tmp_path,
        [
            {
                "module": "darts.pkg.leaky",
                "clause": "zero-consumer",
                "justification": GOOD_JUSTIFICATION,
            }
        ],
    )
    violations, stale, _used = checker.check(tmp_path, allowlist)
    assert flagged(violations) == ["darts.pkg.leaky"]
    assert violations[0].clause == "layering"
    assert any("no longer" in message for message in stale)


def test_allowlist_entry_without_justification_is_rejected(tmp_path):
    write_tree(tmp_path, BASE)
    allowlist = write_allowlist(
        tmp_path, [{"module": "darts.pkg.seed", "clause": "zero-consumer"}]
    )
    with pytest.raises(ValueError, match="missing required field"):
        checker.check(tmp_path, allowlist)


def test_allowlist_rejects_a_token_justification(tmp_path):
    write_tree(tmp_path, BASE)
    allowlist = write_allowlist(
        tmp_path,
        [
            {
                "module": "darts.pkg.seed",
                "clause": "zero-consumer",
                "justification": "n/a",
            }
        ],
    )
    with pytest.raises(ValueError, match="real justification"):
        checker.check(tmp_path, allowlist)


def test_allowlist_rejects_an_unknown_clause(tmp_path):
    write_tree(tmp_path, BASE)
    allowlist = write_allowlist(
        tmp_path,
        [
            {
                "module": "darts.pkg.seed",
                "clause": "vibes",
                "justification": GOOD_JUSTIFICATION,
            }
        ],
    )
    with pytest.raises(ValueError, match="unknown clause"):
        checker.check(tmp_path, allowlist)


def test_allowlist_entry_for_a_deleted_module_is_reported_stale(tmp_path):
    write_tree(
        tmp_path,
        {
            **BASE,
            "darts/pkg/used.py": "VALUE = 1\n",
            "tests/test_used.py": "from darts.pkg.used import VALUE\n",
        },
    )
    allowlist = write_allowlist(
        tmp_path,
        [
            {
                "module": "darts.pkg.gone",
                "clause": "zero-consumer",
                "justification": GOOD_JUSTIFICATION,
            }
        ],
    )
    violations, stale, _used = checker.check(tmp_path, allowlist)
    assert violations == []
    assert any("no such module exists" in message for message in stale)


# --------------------------------------------------------------------------- #
# The real repository
# --------------------------------------------------------------------------- #


def test_shipped_allowlist_is_valid():
    """Every shipped exception carries a clause and a written justification."""
    entries = checker.load_allowlist(
        REPO_ROOT / "helper_scripts" / checker.ALLOWLIST_FILENAME
    )
    assert entries, "the shipped allowlist should not be empty while debt remains"
    for entry in entries:
        assert entry.clause in checker.CLAUSES
        assert len(entry.justification) >= checker.MIN_JUSTIFICATION_LEN


def test_this_repository_satisfies_the_policy():
    """Regression guard: the tree must stay clean once the check is on."""
    if not (REPO_ROOT / "models").is_dir():  # pragma: no cover
        pytest.skip("not a full checkout")
    assert checker.main(["--root", str(REPO_ROOT)]) == 0


def test_layering_clause_holds_with_no_exceptions():
    """No module under darts/ imports from models/, and none is allowlisted."""
    if not (REPO_ROOT / "models").is_dir():  # pragma: no cover
        pytest.skip("not a full checkout")
    _evidence, violations, _scans = checker.analyse(REPO_ROOT)
    assert [v.module for v in violations] == []
