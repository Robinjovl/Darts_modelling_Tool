#!/usr/bin/env python3
"""Enforce the ``darts/`` package placement policy (MR !305 review, section 6).

A module earns its place in the installable ``darts/`` package only if it is
general, has a documented public API, is exercised by at least one in-repo
consumer, and depends only downward. Two of those clauses can be checked
mechanically, and this script checks exactly those two:

``zero-consumer``
    Every module under ``darts/`` must have at least one importer in the
    repository -- another ``darts`` module, a test, a model or a tutorial.
    A module that nothing imports is not exercised by anything, so nothing in
    CI can notice when it breaks. This is the clause that would have caught
    ``darts/pipes/upstream_pressure_node_with_choke.py`` and the four unused
    well plotters before they were shipped inside the wheel.

``layering``
    No module under ``darts/`` may import from ``models/`` or ``tutorials/``.
    The package is the library; the models are its consumers. An upward import
    makes the wheel depend on files that are not in it.

The remaining clauses of the policy -- generality (no case-specific constants,
paths or well names), a documented public API, and the closed condition item
types with ``CustomTerm`` subclasses living model-side -- are review
judgements, not decidable properties, and are deliberately NOT faked here.

Import scanning is AST-based, not textual, so aliased imports
(``import darts.pipes.pipe as p``), conditional imports inside functions or
``try``/``except ImportError`` blocks, and ``__init__`` re-exports are all seen.
Three further kinds of genuine, non-literal consumption are recognised, because
treating them as "no importer" would produce false positives:

1. **Package re-exports.** An import that only appears in the ``__init__.py`` of
   the module's own package is a re-export, not a consumer -- otherwise adding
   one line to ``__init__.py`` would exempt any dead module. The re-exported
   symbols are followed instead: a consumer doing
   ``from darts.nonlinear_solvers import MechanicsNewtonSolver`` (or
   ``darts.nonlinear_solvers.MechanicsNewtonSolver`` on an imported package)
   consumes ``darts/nonlinear_solvers/mechanics.py``.

2. **Dynamic imports and entry points.** ``importlib.import_module("darts.x.y")``
   and any other string literal naming the module, in Python or in the
   configuration files (``pyproject.toml`` entry points, CI YAML), counts.

3. **String-keyed factory registries.** This codebase selects behaviour by name:
   a model passes ``drift_flux_model="tang_2019"`` and never mentions the class
   or the module that implements it. The scan therefore extracts registry keys
   -- string keys of module-level ``{"name": Class}`` dictionaries, string
   arguments of ``register``-style calls, and class-level ``name = "..."``
   markers -- attributes each key to the module that *defines* the registered
   object as well as the one that registers it, and treats the key appearing in
   a consumer as consumption of those modules. Only selector-shaped keys count
   (see :data:`REGISTRY_KEY_RE`), and a registry never vouches for its own
   keys, so listing a dead closure in the table does not keep it alive.

Deliberate exceptions live in ``placement_policy_allowlist.json`` next to this
script, and every entry must carry a written justification.

Usage::

    python helper_scripts/check_placement_policy.py
    python helper_scripts/check_placement_policy.py --verbose
    python helper_scripts/check_placement_policy.py --root /path/to/checkout

Exit status: 0 clean, 1 policy violations, 2 bad invocation or bad allowlist.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: The installable package whose contents the policy governs.
PACKAGE = "darts"

#: Directories that never contain first-party Python we should scan.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".conda-pkgs",
        "__pycache__",
        "build",
        "dist",
        "thirdparty",
        "cmake_patches",
        "open_darts.egg-info",
        ".eggs",
        ".venv",
        "venv",
        "node_modules",
    }
)

#: Top-level directories the package is not allowed to import from. These hold
#: consumers of the library, not parts of it, and they are not in the wheel.
UPWARD_ROOTS = ("models", "tutorials")

#: Directories scanned for consumers but never for policy violations.
#: ``docs`` is deliberately absent: an ``automodule`` directive documents a
#: module, it does not exercise it, so documentation alone must not satisfy the
#: zero-consumer clause.
NON_CONSUMER_DIRS = frozenset({"docs"})

#: Non-Python files scanned textually for module names (entry points, CI).
CONFIG_GLOBS = ("pyproject.toml", "setup.py", ".gitlab-ci.yml", ".cicd/jobs/*.yml")

#: Names of class attributes used in this codebase to declare a registry key.
REGISTRY_NAME_ATTRS = frozenset({"name", "NAME", "registry_key", "key"})

#: Call names that register an object under a string key.
REGISTER_CALL_NAMES = frozenset({"register", "add", "register_class", "register_type"})

#: A registry key must look like a selector name, not like a dictionary field.
#: ``"tang_2019"``, ``"colebrook_white"`` and ``"bhagwat_ghajar_2014"`` qualify;
#: ``"rhs"``, ``"rows"`` and ``"vals"`` -- ordinary data-dict keys -- must not,
#: or every model that pickles a dictionary would vouch for an unrelated module.
REGISTRY_KEY_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)+|[A-Za-z][A-Za-z0-9]{7,}"
)


def is_registry_key(text: str) -> bool:
    """True if ``text`` is distinctive enough to identify a registered module."""
    return 5 <= len(text) <= 64 and REGISTRY_KEY_RE.fullmatch(text) is not None


#: Minimum length of an allowlist justification, so that "n/a" does not pass.
MIN_JUSTIFICATION_LEN = 40

ALLOWLIST_FILENAME = "placement_policy_allowlist.json"

CLAUSES = {
    "zero-consumer": (
        "a module in the installable package must be exercised by at least one "
        "in-repo consumer (another darts module, a test, a model or a tutorial)"
    ),
    "layering": (
        "a module in the installable package must depend only downward; "
        "darts.* may never import from models/ or tutorials/"
    ),
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Violation:
    """One breach of one policy clause."""

    clause: str
    module: str
    path: str
    detail: str

    def render(self) -> str:
        return (
            f"{self.path}: [{self.clause}] {self.module}\n"
            f"    {self.detail}\n"
            f"    policy: {CLAUSES[self.clause]}"
        )


@dataclass
class ModuleScan:
    """Everything the AST pass extracts from one Python file."""

    dotted: str
    path: Path
    is_init: bool = False
    #: Modules this file imports, by dotted name (relative imports resolved),
    #: including the ancestor packages an import also loads.
    imports: set[str] = field(default_factory=set)
    #: The import targets exactly as written, without synthesized ancestors.
    direct_imports: set[str] = field(default_factory=set)
    #: ``from X import n`` pairs where ``X.n`` is not itself a module.
    symbol_imports: set[tuple[str, str]] = field(default_factory=set)
    #: Every string literal in the file.
    strings: set[str] = field(default_factory=set)
    #: Attribute names read off a dotted prefix, as ("darts.pipes", "Pipe").
    attribute_uses: set[tuple[str, str]] = field(default_factory=set)
    #: Registry keys declared in this file, as key -> symbol registered.
    registry_keys: dict[str, str] = field(default_factory=dict)
    #: Class-level ``name = "..."`` markers, as symbol -> key.
    class_name_markers: dict[str, str] = field(default_factory=dict)
    #: Top-level names bound by ``from X import n`` / ``import X as n``.
    symbol_origin: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def iter_python_files(root: Path) -> list[Path]:
    """All first-party ``.py`` files under ``root``, excluding build output."""
    found: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        found.append(path)
    return found


def dotted_name(root: Path, path: Path) -> str:
    """Dotted module name of ``path``; a package ``__init__`` names the package."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


# --------------------------------------------------------------------------- #
# The AST pass
# --------------------------------------------------------------------------- #


def _dotted_of_attribute(node: ast.AST) -> str | None:
    """Render ``a.b.c`` attribute chains back to a dotted string."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class _Visitor(ast.NodeVisitor):
    """Collect imports, strings, registry declarations and attribute uses."""

    def __init__(self, scan: ModuleScan, package_parts: list[str]) -> None:
        self.scan = scan
        self.package_parts = package_parts
        #: local alias -> dotted module it refers to
        self.module_aliases: dict[str, str] = {}
        self._class_stack: list[str] = []
        #: nesting depth; only module-level dict literals can be registries
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    # -- imports ---------------------------------------------------------- #

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.scan.imports.add(alias.name)
            self.scan.direct_imports.add(alias.name)
            # ``import a.b.c`` also makes ``a`` and ``a.b`` load.
            parts = alias.name.split(".")
            for i in range(1, len(parts)):
                self.scan.imports.add(".".join(parts[:i]))
            bound = alias.asname or parts[0]
            self.module_aliases[bound] = alias.name if alias.asname else parts[0]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = self._resolve_from(node)
        if module is None:
            self.generic_visit(node)
            return
        self.scan.imports.add(module)
        self.scan.direct_imports.add(module)
        parts = module.split(".")
        for i in range(1, len(parts)):
            self.scan.imports.add(".".join(parts[:i]))
        for alias in node.names:
            if alias.name == "*":
                self.scan.symbol_imports.add((module, "*"))
                continue
            # ``from pkg import sub`` may be a submodule or a symbol; both are
            # recorded and disambiguated later against the module inventory.
            self.scan.symbol_imports.add((module, alias.name))
            bound = alias.asname or alias.name
            self.scan.symbol_origin[bound] = module
            self.module_aliases[bound] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def _resolve_from(self, node: ast.ImportFrom) -> str | None:
        if not node.level:
            return node.module
        # Relative import: walk up from the containing package.
        base = list(self.package_parts)
        up = node.level - 1
        if up:
            if up > len(base):
                return None
            base = base[: len(base) - up]
        if node.module:
            base = base + node.module.split(".")
        return ".".join(base) if base else None

    # -- strings and attribute uses ---------------------------------------- #

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.scan.strings.add(node.value)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        dotted = _dotted_of_attribute(node)
        if dotted is not None:
            head, _, attr = dotted.rpartition(".")
            root_alias, _, tail = head.partition(".")
            target = self.module_aliases.get(root_alias)
            if target is not None:
                prefix = target if not tail else f"{target}.{tail}"
                self.scan.attribute_uses.add((prefix, attr))
        self.generic_visit(node)

    # -- registry declarations --------------------------------------------- #

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for stmt in node.body:
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(stmt, ast.Assign):
                targets, value = list(stmt.targets), stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets, value = [stmt.target], stmt.value
            if value is None or not isinstance(value, ast.Constant):
                continue
            if not isinstance(value.value, str) or not value.value:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in REGISTRY_NAME_ATTRS
                    and is_registry_key(value.value)
                ):
                    self.scan.class_name_markers[node.name] = value.value
        self._class_stack.append(node.name)
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1
        self._class_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        self._collect_registry_dict(node.value)
        # ``REGISTRY["key"] = Cls``
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
                and is_registry_key(target.slice.value)
            ):
                symbol = self._symbol_of(node.value)
                if symbol is not None:
                    self.scan.registry_keys[target.slice.value] = symbol
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        fname = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        )
        if fname in REGISTER_CALL_NAMES and len(node.args) >= 2:
            key, obj = node.args[0], node.args[1]
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and is_registry_key(key.value)
            ):
                symbol = self._symbol_of(obj)
                if symbol is not None:
                    self.scan.registry_keys[key.value] = symbol
        self.generic_visit(node)

    def _collect_registry_dict(self, value: ast.expr) -> None:
        """Record module-level ``{"key": Class}`` literals as a registry.

        Only module level, and only distinctive keys: a data dictionary built
        inside a function (``jac = {"rows": ..., "rhs": ...}``) is not a
        registry, and treating it as one would let any model that mentions
        ``"rhs"`` vouch for the module that built it.
        """
        if self._depth or not isinstance(value, ast.Dict) or not value.keys:
            return
        pairs: list[tuple[str, str]] = []
        for key, item in zip(value.keys, value.values, strict=False):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return
            if not is_registry_key(key.value):
                return
            symbol = self._symbol_of(item)
            if symbol is None:
                return
            pairs.append((key.value, symbol))
        for key_value, symbol in pairs:
            self.scan.registry_keys[key_value] = symbol

    @staticmethod
    def _symbol_of(node: ast.expr) -> str | None:
        """The bare name of a referenced class or function, if it is one."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


def scan_python_file(root: Path, path: Path) -> ModuleScan | None:
    """Parse one file; return ``None`` if it is not parseable Python."""
    dotted = dotted_name(root, path)
    scan = ModuleScan(
        dotted=dotted,
        path=path,
        is_init=path.name == "__init__.py",
    )
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            # Parsing surfaces SyntaxWarning/DeprecationWarning for invalid
            # escapes in third-party-style sources; those are somebody else's
            # defect, not a placement-policy finding.
            warnings.simplefilter("ignore")
            tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError, OSError):
        return None
    package_parts = dotted.split(".") if scan.is_init else dotted.split(".")[:-1]
    _Visitor(scan, package_parts).visit(tree)
    return scan


# --------------------------------------------------------------------------- #
# Consumption analysis
# --------------------------------------------------------------------------- #


@dataclass
class Evidence:
    """Why a module is considered consumed."""

    kind: str
    source: str

    def render(self) -> str:
        return f"{self.kind} ({self.source})"


def _is_ancestor_init(scan: ModuleScan, module: str) -> bool:
    """True if ``scan`` is the ``__init__`` of a package containing ``module``."""
    return scan.is_init and module.startswith(scan.dotted + ".")


def build_reexport_map(
    scans: dict[str, ModuleScan], modules: set[str]
) -> dict[tuple[str, str], str]:
    """Map ``(package, symbol)`` re-exported by an ``__init__`` to its module."""
    reexports: dict[tuple[str, str], str] = {}
    for scan in scans.values():
        if not scan.is_init:
            continue
        for source_module, symbol in scan.symbol_imports:
            if symbol == "*":
                continue
            if f"{source_module}.{symbol}" in modules:
                # ``from .sub import name`` where name is itself a module.
                reexports[(scan.dotted, symbol)] = f"{source_module}.{symbol}"
            elif source_module in modules:
                reexports[(scan.dotted, symbol)] = source_module
    # Follow one more hop, so a symbol re-exported by a nested __init__ and then
    # by the parent __init__ still resolves to the module that defines it.
    for _ in range(3):
        changed = False
        for (package, symbol), target in list(reexports.items()):
            deeper = reexports.get((target, symbol))
            if deeper is not None and deeper != target:
                reexports[(package, symbol)] = deeper
                changed = True
        if not changed:
            break
    return reexports


def build_registry_index(
    scans: dict[str, ModuleScan], modules: set[str], root: Path
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Index the string-keyed factory registries of the package.

    Returns ``(key -> modules the key exercises, key -> files declaring it)``.

    A registry host such as ``darts/pipes/drift_flux.py`` maps ``"tang_2019"``
    to ``Tang2019Closure``. If that class was imported from another module, the
    key exercises the module that *defines* it as well as the module that
    registers it -- which is the case the policy has to get right when the
    closures live in their own files and are only ever selected by name.

    The declaring files are tracked separately because they must not count as
    users of their own key: a registry listing every closure it knows about
    would otherwise vouch for all of them, dead ones included.
    """
    index: dict[str, set[str]] = {}
    declared_in: dict[str, set[str]] = {}
    for scan in scans.values():
        if not scan.dotted.startswith(PACKAGE):
            continue
        rel = str(scan.path.relative_to(root))
        for key, symbol in scan.registry_keys.items():
            origin = scan.symbol_origin.get(symbol)
            owners = {scan.dotted}
            if origin is not None:
                candidate = f"{origin}.{symbol}"
                if candidate in modules:
                    owners.add(candidate)
                elif origin in modules:
                    owners.add(origin)
            index.setdefault(key, set()).update(owners)
            declared_in.setdefault(key, set()).add(rel)
        for symbol, key in scan.class_name_markers.items():
            del symbol
            index.setdefault(key, set()).add(scan.dotted)
            declared_in.setdefault(key, set()).add(rel)
    return index, declared_in


def analyse(
    root: Path, verbose: bool = False
) -> tuple[
    dict[str, list[Evidence]],
    list[Violation],
    dict[str, ModuleScan],
]:
    """Scan the tree and return per-module evidence plus layering violations."""
    py_files = iter_python_files(root)
    scans: dict[str, ModuleScan] = {}
    for path in py_files:
        scan = scan_python_file(root, path)
        if scan is None:
            continue
        scans[str(path.relative_to(root))] = scan

    package_modules = {
        scan.dotted
        for scan in scans.values()
        if scan.dotted == PACKAGE or scan.dotted.startswith(PACKAGE + ".")
    }
    by_dotted = {
        scan.dotted: scan for scan in scans.values() if scan.dotted in package_modules
    }
    reexports = build_reexport_map(scans, package_modules)
    registry, key_declarers = build_registry_index(scans, package_modules, root)

    consumer_scans = [
        scan
        for rel, scan in scans.items()
        if Path(rel).parts[0] not in NON_CONSUMER_DIRS
    ]

    evidence: dict[str, list[Evidence]] = {module: [] for module in package_modules}

    # -- 1. literal imports ------------------------------------------------ #
    for scan in consumer_scans:
        rel = str(scan.path.relative_to(root))
        targets = set(scan.imports)
        for source_module, symbol in scan.symbol_imports:
            candidate = f"{source_module}.{symbol}"
            if candidate in package_modules:
                targets.add(candidate)
        for target in sorted(targets):
            if target not in package_modules or target == scan.dotted:
                continue
            if _is_ancestor_init(scan, target):
                # Re-export, not consumption; followed separately below.
                continue
            evidence[target].append(Evidence("imported", rel))

    # -- 2. package re-exports actually used ------------------------------- #
    for scan in consumer_scans:
        if scan.is_init and scan.dotted.startswith(PACKAGE):
            continue  # an __init__ using its own re-export proves nothing
        rel = str(scan.path.relative_to(root))
        used: set[tuple[str, str]] = set(scan.symbol_imports) | scan.attribute_uses
        for package, symbol in used:
            target = reexports.get((package, symbol))
            if target is None or target == scan.dotted:
                continue
            evidence[target].append(
                Evidence("used via re-export", f"{rel}: {package}.{symbol}")
            )

    # -- 3. dynamic imports and string references -------------------------- #
    literals: dict[str, set[str]] = {}
    for scan in consumer_scans:
        rel = str(scan.path.relative_to(root))
        for text in scan.strings:
            literals.setdefault(text, set()).add(rel)
    for text in _config_strings(root):
        for module in package_modules:
            if module in text:
                literals.setdefault(module, set()).add("configuration")

    for module in package_modules:
        sources = set(literals.get(module, ()))
        for text, where in literals.items():
            if (
                text.startswith(module + ":")
                or text == module.replace(".", "/") + ".py"
            ):
                sources |= where
        for where in sorted(sources):
            if where != str(by_dotted[module].path.relative_to(root)):
                evidence[module].append(Evidence("named dynamically", where))

    # -- 4. string-keyed factory registries -------------------------------- #
    for key, owners in registry.items():
        users = set(literals.get(key, ())) - key_declarers.get(key, set())
        if not users:
            continue
        for owner in owners:
            if owner not in evidence:
                continue
            for where in sorted(users):
                evidence[owner].append(
                    Evidence("selected by registry key", f"{where}: {key!r}")
                )

    # -- layering ---------------------------------------------------------- #
    upward_paths = _upward_module_names(root)
    violations: list[Violation] = []
    for module, scan in sorted(by_dotted.items()):
        offenders = sorted(
            target
            for target in scan.direct_imports
            if _is_upward(target, upward_paths, module, package_modules)
        )
        for target in offenders:
            violations.append(
                Violation(
                    clause="layering",
                    module=module,
                    path=str(scan.path.relative_to(root)),
                    detail=(
                        f"imports {target!r}, which lives outside the installable "
                        f"package (under {'/ or '.join(UPWARD_ROOTS)}/)"
                    ),
                )
            )

    if verbose:
        for module in sorted(evidence):
            marks = evidence[module]
            status = "OK " if marks else "!! "
            print(f"{status}{module}")
            for mark in marks[:6]:
                print(f"      {mark.render()}")
            if len(marks) > 6:
                print(f"      ... and {len(marks) - 6} more")

    return evidence, violations, by_dotted


def _config_strings(root: Path) -> list[str]:
    """Raw text of the configuration files that can name a module."""
    texts: list[str] = []
    for pattern in CONFIG_GLOBS:
        for path in sorted(root.glob(pattern)):
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return texts


def _upward_module_names(root: Path) -> set[str]:
    """Top-level import names that would resolve into ``models/``/``tutorials/``.

    ``models/2ph_comp_solid/model.py`` makes a bare ``import model`` inside
    ``darts/`` a layering violation just as much as ``import models.foo`` is,
    and the bare form is the one that actually happens.
    """
    names: set[str] = set()
    for top in UPWARD_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        names.add(top)
        for path in base.rglob("*.py"):
            if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
                continue
            if path.name == "__init__.py":
                names.add(path.parent.name)
            else:
                names.add(path.stem)
    # Never claim a name the package itself provides.
    names.discard(PACKAGE)
    return names


def _is_upward(
    target: str,
    upward_names: set[str],
    importer: str,
    package_modules: set[str],
) -> bool:
    """True if ``importer`` (a darts module) reaches outside the package.

    A bare ``import shapes`` is only upward if it cannot mean a sibling module
    of the importer or a top-level module of the package: several old modules
    under ``darts/reservoirs/mesh/geometry/`` still use implicit-relative
    imports of their own siblings, and those are a different defect.
    """
    head = target.split(".")[0]
    if head in UPWARD_ROOTS:
        return True
    if head not in upward_names:
        return False
    package = importer.rpartition(".")[0]
    sibling = f"{package}.{head}" if package else head
    if sibling in package_modules or f"{PACKAGE}.{head}" in package_modules:
        return False
    return True


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #


@dataclass
class AllowEntry:
    module: str
    clause: str
    justification: str


def load_allowlist(path: Path) -> list[AllowEntry]:
    """Read and validate the allowlist; raise ``ValueError`` on a bad entry."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or "allow" not in raw:
        raise ValueError(f"{path}: expected an object with an 'allow' array")
    entries: list[AllowEntry] = []
    for index, item in enumerate(raw["allow"]):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: allow[{index}] is not an object")
        missing = [k for k in ("module", "clause", "justification") if k not in item]
        if missing:
            raise ValueError(
                f"{path}: allow[{index}] is missing required field(s) "
                f"{', '.join(missing)}; every exception must name the module, "
                f"the policy clause it is exempt from, and why"
            )
        clause = item["clause"]
        if clause not in CLAUSES:
            raise ValueError(
                f"{path}: allow[{index}] has unknown clause {clause!r}; "
                f"expected one of {', '.join(sorted(CLAUSES))}"
            )
        justification = str(item["justification"]).strip()
        if len(justification) < MIN_JUSTIFICATION_LEN:
            raise ValueError(
                f"{path}: allow[{index}] ({item['module']}) needs a real "
                f"justification of at least {MIN_JUSTIFICATION_LEN} characters, "
                f"got {len(justification)}"
            )
        entries.append(
            AllowEntry(
                module=str(item["module"]),
                clause=clause,
                justification=justification,
            )
        )
    return entries


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def check(
    root: Path, allowlist_path: Path, verbose: bool = False
) -> tuple[list[Violation], list[str], list[AllowEntry]]:
    """Run both clauses. Returns (violations, stale allowlist entries, used)."""
    entries = load_allowlist(allowlist_path)
    evidence, violations, by_dotted = analyse(root, verbose=verbose)

    for module in sorted(evidence):
        if evidence[module]:
            continue
        scan = by_dotted[module]
        if scan.is_init:
            # A package __init__ exists so the package is importable at all; it
            # is judged through the modules it contains, not on its own.
            continue
        violations.append(
            Violation(
                clause="zero-consumer",
                module=module,
                path=str(scan.path.relative_to(root)),
                detail=(
                    "no importer anywhere in the repository -- not another darts "
                    "module, not a test, not a model, not a tutorial; and it is "
                    "not named dynamically or selected by a registry key either"
                ),
            )
        )

    allowed = {(entry.module, entry.clause): entry for entry in entries}
    remaining: list[Violation] = []
    used: list[AllowEntry] = []
    for violation in violations:
        entry = allowed.get((violation.module, violation.clause))
        if entry is None:
            remaining.append(violation)
        else:
            used.append(entry)

    known = set(evidence)
    stale: list[str] = []
    for entry in entries:
        if entry.module not in known:
            stale.append(
                f"{entry.module}: allowlisted for '{entry.clause}' but no such "
                f"module exists under {PACKAGE}/ -- remove the entry"
            )
        elif entry not in used:
            stale.append(
                f"{entry.module}: allowlisted for '{entry.clause}' but no longer "
                f"violates it -- remove the entry"
            )
    return remaining, stale, used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check the darts/ package placement policy: no module without an "
            "in-repo consumer, and no upward import from models/."
        )
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repository root to check (default: the checkout this script is in)",
    )
    parser.add_argument(
        "--allowlist",
        default=None,
        help=f"Allowlist file (default: helper_scripts/{ALLOWLIST_FILENAME})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the consumption evidence found for every module",
    )
    parser.add_argument(
        "--strict-stale",
        action="store_true",
        help="Fail when the allowlist has entries that are no longer needed",
    )
    # Accept and ignore filenames, so the hook can be run either way.
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / PACKAGE).is_dir():
        print(f"error: {root} does not contain a {PACKAGE}/ package", file=sys.stderr)
        return 2
    allowlist_path = (
        Path(args.allowlist)
        if args.allowlist
        else Path(__file__).resolve().parent / ALLOWLIST_FILENAME
    )

    try:
        violations, stale, used = check(root, allowlist_path, verbose=args.verbose)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if used and args.verbose:
        print("\nAllowed by exception:")
        for entry in used:
            print(f"  {entry.module} [{entry.clause}]: {entry.justification}")

    for message in stale:
        print(f"warning: stale allowlist entry: {message}", file=sys.stderr)

    if violations:
        print(
            f"\nPlacement policy: {len(violations)} violation(s) "
            f"(MR !305 review, section 6)\n",
            file=sys.stderr,
        )
        for violation in sorted(violations, key=lambda v: (v.clause, v.module)):
            print(violation.render() + "\n", file=sys.stderr)
        print(
            "Fix by giving the module a consumer (a CI-wired model or a test), "
            "moving it out of the installable package, or -- if it is a "
            "deliberate library seed -- adding it to "
            f"helper_scripts/{ALLOWLIST_FILENAME} with a written justification.",
            file=sys.stderr,
        )
        return 1

    if stale and args.strict_stale:
        return 1

    if used:
        modules = sorted({entry.module for entry in used})
        print(
            f"Placement policy: OK ({len(modules)} module(s) allowed by written "
            f"exception in helper_scripts/{ALLOWLIST_FILENAME}: "
            f"{', '.join(modules)})"
        )
    else:
        print("Placement policy: OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
