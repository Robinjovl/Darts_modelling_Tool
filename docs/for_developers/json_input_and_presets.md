# JSON input system and the preset registry

A single document describing the JSON-driven model specification layer
(`darts.api`), the preset registry that sits on top of it, and how the
two fit together with the legacy Python-driven `model.py` workflow.

Status: post-MR !300 plus the preset programme tracked in
[`preset_plan.md`](json_input_and_presets.md#status-and-remaining-work).

---

## 1. Motivation — why a JSON input layer

Before this work, every DARTS model was a hand-written Python class
that inherited from `DartsModel`, overrode `set_input_data`,
`set_reservoir`, `set_physics`, `set_wells`, `set_initial_conditions`,
`set_well_controls` and `set_sim_params`, and threaded its parameters
through a free-form `InputData` god-struct. That works well for an
expert author, but it has three structural problems:

1. **Non-serializable.** There is no canonical text representation of
   "the model" that a tool, an LLM agent, an MCP server, or a CI job
   can read, validate, and act on.
2. **No schema, no defaults policy.** Required vs. optional fields
   are implicit in `InputData.check()` logic; missing values surface
   as Python `AttributeError` deep in `init()`.
3. **No composition.** Two models that share a fluid system or a
   numerics setup still copy-paste the relevant blocks.

The JSON input layer (`darts.api`) fixes the first two problems by
making every section a Pydantic-validated `Config`. The preset
registry (`darts.api.presets`) fixes the third by letting any of those
configs ship as a named, version-tagged, composable JSON file.

---

## 2. The two entry points

There are now two equivalent ways to specify and run a model — both
hit the same builder and produce the same in-memory `DartsModel`:

### 2.1 Single `ModelSpec` JSON

One file, one section per concern. Pass through the CLI or the
`ModelBuilder.apply_dict()` Python API:

```bash
darts --json models/2ph_comp/2ph_comp.json --days 1000 --report-days 100 --vtk-output
```

```python
import json
from darts.api import ModelBuilder
from darts.api.json_model import JsonModel

with open("model.json") as fp:
    spec_dict = json.load(fp)

model = JsonModel()
ModelBuilder.apply_dict(spec_dict, model)   # resolves section presets + validates + applies
model.init()
model.run(...)
```

`apply_dict()` is the canonical single-JSON entry point: it runs
`resolve_section_presets()` to expand any `{"preset": "..."}` shorthand
in `reservoir`, `physics`, `sim_params`, … and *then* validates against
`StrictModelSpec`.  The two-step alternative
(`ModelSpec.model_validate(spec_dict)` → `ModelBuilder.apply(spec, model)`)
also works, but skipping the resolution pre-pass means the modular
preset composition idiom rejects with `extra="forbid"` because
`{"preset": "..."}` is not a valid PhysicsSpec/ReservoirSpec key.  The
following entry points all run the pre-pass for you:

* CLI: `darts --json model.json` (via `darts.api.run_json_model`)
* `darts.api.ModelBuilder.apply_dict(spec_dict, model)`
* `darts.api.JsonModelAdapter.apply_spec_dict(spec_dict)`
* `darts.api.MCPModelAdapter.build_model()`
* `darts.models.DartsModel.apply_model_spec(spec_dict)`
* `models/json_test_suite.py` (regression harness)

### 2.2 Step-by-step (MCP server)

The same `ModelSpec` schema, but the sections are committed one tool
call at a time:
`create_model`, `set_reservoir`, `set_physics`, `set_wells`,
`set_initial_conditions`, `set_well_controls`, `set_sim_params`,
`run_model`. Each request is a JSON object matching the corresponding
`PatchModelSpec.<section>` subschema (RFC-7396 merge-patch
semantics — missing fields keep their previous value).

The MCP server lives in a separate repo (`mcp-server-langchain`) and
talks to DARTS only via these tools, so it's safe to drive from a
LangGraph agent without giving the LLM raw Python access.

---

## 3. `ModelSpec` schema

Top-level shape (`darts.api.schemas.StrictModelSpec`):

```jsonc
{
  "plugin_registry": { /* optional — user-defined code plugins */ },
  "reservoir":      { "type": "structured" | "cpg", ... },
  "physics":        { "plugin": {...}, "components": [...],
                      "phases": [...], "property_regions": [...] },
  "wells":          { "wells": [ {"name": "...", "perforations": [...]} ] },
  "initial_conditions": { "by_array": { "pressure": ..., "<comp>": ... } },
  "well_controls":  { /* top-level defaults; per-well overrides live in wells.* */ },
  "sim_params":     { "first_ts": ..., "runtime": ..., "tol_newton": ... },
  "output":         { "folder": "...", "precision": "d" }
}
```

Every section is **its own native `Config` class** living next to the
implementation, with `extra="forbid"` so typos fail fast:

| Section | Native Config (and module) |
|---|---|
| `reservoir.structured` | `StructReservoirConfig` (`darts/reservoirs/struct_reservoir.py`) |
| `reservoir.cpg` | `CPGReservoirConfig` (`darts/reservoirs/cpg_reservoir.py`) |
| `physics` (Compositional) | `CompositionalConfig` (`darts/physics/super/physics.py`) |
| `physics` (BlackOil) | `BlackOilConfig` (`darts/physics/blackoil.py`) |
| `sim_params` | `SimParamsConfig` (`darts/models/darts_model.py`) |
| property containers, evaluator plugins | `PropertyContainerConfig`, `ConstFuncConfig`, `PhaseRelPermConfig`, `DensityBasicConfig`, `EnthalpyBasicConfig`, `ConstantKConfig`, `KineticBasicConfig`, … (`darts/physics/properties/*` and `darts/physics/super/*`) |

`darts.api.schemas` re-exports each of these under a `Strict<Name>Spec`
alias (e.g. `StrictReservoirSpec`, `StrictSimParamsSpec`) so existing
imports keep working — but the canonical Config now lives next to its
runtime class.

Each native class also exposes a `from_config(cfg)` classmethod:
`StructReservoir.from_config(cfg, timer=…)`, `Compositional.from_config(cfg)`,
`PhaseRelPerm.from_config(cfg)`, etc. The builder is therefore a thin
orchestrator — it does not duplicate construction logic.

---

## 4. Builder pipeline

```
JSON / dict
   │
   ▼  Pydantic validation (extra="forbid", required-field checks)
StrictModelSpec
   │
   ▼  $preset refs resolved (if any) + DataRef refs resolved to plain values
   │  + plugin_registry entries loaded from constructor paths
   ▼
ModelBuilder.apply(spec, model)
   │
   ├─► model.set_reservoir(...)        → StructReservoir.from_config(...)
   ├─► model.set_physics(...)          → Compositional.from_config(...)
   ├─► model.set_wells_from_dict(...)
   ├─► model.set_initial_conditions_from_dict(...)
   ├─► model.set_well_controls_from_dict(...)
   └─► model.set_sim_params_from_config(...)  → DartsModel.set_sim_params(...)
   │
   ▼  user-driven
model.init() ; model.run(...) ; model.output_* / VTK render
```

Two important builder behaviours:

* **Layered reservoir expansion.** `ModelSpec.reservoir.layers` (a
  list of `ReservoirLayerConfig` blocks with per-layer `count`,
  `permx`, `perm…`, `poro`, `depth`) is expanded into flat per-cell
  arrays inside `StructReservoir.from_config()`. Python users get the
  same convenience without going through the JSON path.
* **DataRef widening.** Inside `darts.api`, reservoir property fields
  (`dx`, `dy`, `dz`, `permx`, …, `depth`) accept
  `ReservoirValue = ScalarOrArray | DataRef`. A `DataRef` points at
  an external file or a registered in-memory object; the builder
  resolves it to a plain value before handing it to the native
  `StructReservoirConfig` (which only knows `ScalarOrArray`). This
  keeps the native config simple while the JSON-transport layer
  supports out-of-tree data.

---

## 5. Preset registry — concept

A **preset** is a small JSON file holding metadata plus a *partial*
`ModelSpec` payload — typically one section. It lives in a
hierarchical directory tree under the canonical envelope:

```jsonc
{
  "_meta": {
    "name":        "co2_brine_isothermal",
    "description": "Isothermal CO2-C1-brine two-phase compositional physics …",
    "references":  ["Spycher & Pruess (2005) — Geochim. Cosmochim. Acta, 69(13)"],
    "tags":        ["physics", "compositional", "co2", "brine", "isothermal", "co2-storage"],
    "version":     "1"
  },
  "config":              { /* the Pydantic Config payload */ },
  "property_regions":    [ /* optional — physics presets that ship a full evaluator stack */ ],
  "plugin_registry":     { /* optional — presets that ship code plugins */ }
}
```

### 5.1 Anatomy

| Block | What it holds | Validated against |
|---|---|---|
| `_meta` | self-describing metadata for discovery (`name`, `description`, `references`, `tags`, `version`) | `darts.api.presets.PresetMeta` |
| `config` | a Pydantic-validated Config payload (e.g. `CompositionalConfig`, `StructReservoirConfig`, `SimParamsConfig`, `ConstantKConfig` …) | the Config class bound to the preset's directory or to the `kind` discriminator inside `config` |
| `property_regions` (optional) | a list of property-region blocks (`property_container` + `plugins` dicts) for physics presets that want to ship a full evaluator stack | not Pydantic-validated at the preset layer — validated downstream by `StrictPhysicsSpec` when the patch is built |
| `plugin_registry` (optional) | a `plugin_registry` block declaring code plugins the preset ships itself; relative `constructor` paths are resolved against the preset file's directory at load time | `PluginRegistrySpec` |

### 5.2 Where presets live

The default tree is **shipped as package data** under
`darts/api/presets_data/` so wheel installs work without source-tree
access. The pyproject glob `"darts.api" = ["presets_data/**/*.json",
"presets_data/**/*.py"]` ships every preset (and any bundled code
plugin under `presets_data/code/`).

The loader's search precedence:

1. `DARTS_PRESET_ROOT` env var — first-priority root; useful for an
   out-of-tree preset library or a per-project override.
2. `DEFAULT_PRESET_ROOT` — resolved via `importlib.resources.files(
   "darts.api") / "presets_data"`, so it works for both editable and
   wheel installs.
3. Anything registered at runtime by `load_preset_dir()` or
   `register_preset()`.

### 5.3 Composition via `$preset` references

Any nested dict of the form `{"$preset": "<qualified-name>"}` is
replaced at load time by the referenced preset's `config` payload.
Cycles are detected and rejected; cross-file references are resolved
lazily (no fixed load order).

Example — the shipped `physics/compositional/co2_brine_isothermal`
preset composes its flash and rel-perm plugins from other presets:

```jsonc
"plugins": {
  "flash_ev": {
    "type_id": "flash/ConstantK@v1",
    "config":  { "$preset": "evaluators/flash/constant_k_co2_c1_brine" }
  },
  "rel_perm_ev": {
    "gas": { "type_id": "relperm/PhaseRelPerm@v1",
             "config":  { "$preset": "evaluators/rel_perm/quadratic_gas" } },
    "oil": { "type_id": "relperm/PhaseRelPerm@v1",
             "config":  { "$preset": "evaluators/rel_perm/quadratic_oil" } }
  }
}
```

The same `$preset` ref also works inside the top-level `property_regions`
block of a physics preset.

### 5.4 Directory bindings

Evaluator presets dispatch through the existing `kind` discriminator
on `EvaluatorConfigBase` (e.g. `"kind": "density_basic"`), so they
need no separate binding. Non-evaluator categories bind their
top-level directory to a Config class explicitly via
`register_preset_directory_binding()` — done at package import for the
shipped categories:

| Directory | Config class |
|---|---|
| `property_container/` | `PropertyContainerConfig` |
| `physics/compositional/` | `CompositionalConfig` |
| `physics/black_oil/` | `BlackOilConfig` |
| `physics/dead_oil/` | `CompositionalConfig` (dead-oil is dispatched through the Compositional plugin family) |
| `reservoirs/structured/` | `StructReservoirConfig` |
| `sim_params/` | `SimParamsConfig` |

Adding a new category is a single call in
`darts/api/presets.py::_register_default_directory_bindings()`.

---

## 6. Shipped preset catalog

The repository ships **37 presets** under `darts/api/presets_data/`:

| Category | Count | Selected entries |
|---|---|---|
| `evaluators/density/` | 5 | `co2_brine`, `spivey_2004_brine`, `garcia_2001_co2_brine`, `co2_aquifer_enriched`, `spe1_oil` |
| `evaluators/viscosity/` | 3 | `water_constant`, `co2_supercritical`, `oil_light` |
| `evaluators/enthalpy/` | 2 | `water_basic`, `co2_basic` |
| `evaluators/rel_perm/` | 4 | `quadratic_water`, `quadratic_oil`, `quadratic_gas`, `van_genuchten_water` |
| `evaluators/capillary_pressure/` | 2 | `brooks_corey_2ph`, `van_genuchten_2ph` |
| `evaluators/flash/` | 4 | `constant_k_co2_brine`, `constant_k_co2_c1_brine`, `single_phase`, `solid_flash_co2_brine_calcite` (uses `$preset` itself to wrap an inner flash) |
| `evaluators/iapws/` | 2 | `water_density_ph`, `steam_density_ph` |
| `evaluators/kinetic/` | 1 | `calcite_dissolution_simple` |
| `evaluators/rock/` | 1 | `standard_compaction` |
| `property_container/` | 3 | `co2_brine_2ph`, `spe1_3ph`, `water_thermal` |
| `physics/compositional/` | 2 | `co2_brine_isothermal` *(self-contained; ships `property_regions`)*, `general_2ph_pt` |
| `physics/black_oil/` | 1 | `spe1_isothermal` |
| `physics/dead_oil/` | 1 | `cpg_deadoil_brugge` *(self-contained; ships `property_regions` AND `plugin_registry` with a bundled `ModelProperties` code plugin under `presets_data/code/dead_oil_model_properties.py`)* |
| `reservoirs/structured/` | 3 | `spe1_uniform`, `quarter_5spot_2d`, `layered_caprock` |
| `sim_params/` | 3 | `default_implicit`, `fast_explicit_screening`, `long_storage_simulation` |

Bundled code at `darts/api/presets_data/code/` (currently:
`dead_oil_model_properties.py`) lets a preset ship a custom property
container that resolves to an absolute path at load time, so the
preset is self-contained in wheel installs.

---

## 7. Using a preset

### 7.1 In a single `ModelSpec` JSON

Anywhere a section accepts a Pydantic Config, the **preset name** can
be passed instead and the loader substitutes the resolved config in.
The most reusable shape today is to put the preset reference on a
section's top-level dict alongside any user overrides — the loader
patch-merges the request fields onto the preset baseline:

```jsonc
{
  "reservoir":  { "preset": "reservoirs/structured/spe1_uniform" },
  "physics":    { "preset": "physics/compositional/co2_brine_isothermal" },
  "wells":      { "wells": [...] },
  "initial_conditions": { "by_array": {"pressure": 50, "CO2": 0.1, "C1": 0.2} },
  "well_controls": { "inj_bhp": 140, "prod_bhp": 40, "inj_composition": [0.99999998, 1e-8] },
  "sim_params": { "preset": "sim_params/default_implicit",
                  "runtime": 100 }                /* override just the duration */
}
```

### 7.2 From Python

```python
from darts.api.presets import load_preset
p = load_preset("physics/compositional/co2_brine_isothermal")
print(p.meta.tags)            # ['physics', 'compositional', 'co2', 'brine', ...]
print(p.property_regions)     # full evaluator stack (with $preset refs resolved)
phys_cfg = p.config           # validated CompositionalConfig instance
```

### 7.3 From the MCP server

The MCP tool requests (`set_physics`, `set_reservoir`, `set_sim_params`)
all accept a top-level `preset` field with the same semantics:

```json
{"name": "set_physics",
 "arguments": {"request": {"preset": "physics/compositional/co2_brine_isothermal"}}}
```

When a preset ships its own `property_regions` (flagged by
`p.property_regions is not None` and surfaced as
"[self-contained — ships property_regions]" in graph prompts),
no further parameters are needed; the server's
`InteractiveModel.build_physics_patch()` carries the preset payload
straight into a `StrictPhysicsSpec`. When a preset also ships its own
`plugin_registry` (e.g. `physics/dead_oil/cpg_deadoil_brugge`), the
constructor paths are already absolute-path-resolved at load time so
wheel users do not need `OPEN_DARTS_MODELS_ROOT`.

---

## 8. Are presets used in the shipped model JSONs?

**Not yet.** This is worth being explicit about because the answer is
counter-intuitive.

The reference models under `models/2ph_comp/`, `models/2ph_do/`,
`models/2ph_do_thermal/`, `models/3ph_bo/`, `models/3ph_comp_w/`,
`models/3ph_do/`, `models/2ph_comp_solid/`, and
`models/cpg_deadoil_brugge/` each ship a fully-inline JSON
(`<name>/<name>.json`) — every density, viscosity, rel-perm and
flash evaluator is spelled out under `physics.property_regions[].plugins`,
not referenced. Search confirms:

```bash
grep -rln '"preset"\|"\$preset"' --include='*.json' models/
# → no matches
```

By contrast, **the shipped presets themselves DO use `$preset` composition**:

```bash
grep -rln '"\$preset"' --include='*.json' darts/api/presets_data/
# → darts/api/presets_data/evaluators/flash/solid_flash_co2_brine_calcite.json
#   darts/api/presets_data/physics/compositional/co2_brine_isothermal.json
```

So the *building blocks* are composable, but the *reference fixtures*
aren't yet rewritten to use them. There are two reasons for that:

1. **Regression PKLs.** Every `models/<id>/ref/*.pkl` was generated
   from the inline JSON; converting a reference model to a preset
   reference means re-running its full PKL suite (and accepting the
   exact-equality bar that `json_test_suite.py` enforces). It's
   mechanical but needs care.
2. **Closed `ModelType` enum.** The MCP server still ships seven
   prompt templates (`RUN_2PH_COMP_PROMPT_TEXT` …) that step-by-step
   reproduce each reference model with `{"model_type": "<id>"}` in
   step 3. Those prompts pre-date the preset registry; they will be
   regenerated against parity presets in the next pass.

The cross-references are recorded today on each `READY_MODELS` entry
in `open-darts-mcp/server/subservers/prompts.py` (a `preset_id`
field). Entries that already have a parity preset:

| Legacy `model_type` | Parity preset |
|---|---|
| `2ph_comp` | `physics/compositional/co2_brine_isothermal` |
| `3ph_bo` | `physics/black_oil/spe1_isothermal` |
| `cpg_deadoil_brugge` | `physics/dead_oil/cpg_deadoil_brugge` |
| `2ph_do`, `2ph_do_thermal`, `3ph_do`, `3ph_comp_w` | (pending) |

The eventual end-state — each reference JSON shrinks to a few lines
(`{"preset": "<...>"}` per section plus a `wells` and
`initial_conditions` block) — is straightforward once the four
pending parity presets land.

---

## 9. Plug-in registry — user code from JSON

Two ways a user can plug their own code into the JSON workflow:

### 9.1 Top-level `plugin_registry` on `ModelSpec`

The classical path. Each entry binds a `type_id` to a constructor
string `"module.py:ClassName"`. Constructors can live anywhere on the
filesystem; relative paths resolve against the JSON file's directory.

```jsonc
"plugin_registry": {
  "entries": [
    {
      "type_id":      "pc/MyPropertyContainer@v1",
      "kind":         "pc",
      "constructor":  "./model.py:MyPropertyContainer",
      "config_model": "darts.api.type_registry:PropertyContainerConfig",
      "customizable": true,
      "doc":          "Per-model property container"
    }
  ]
}
```

Used by `models/cpg_deadoil_brugge/cpg_deadoil_brugge.json` to
reference a custom `ModelProperties` shipped alongside the JSON.

### 9.2 Preset-owned `plugin_registry`

When a code plugin is reusable across cases, it can live inside a
preset's directory and be carried automatically with the preset.
`physics/dead_oil/cpg_deadoil_brugge` does this:

```jsonc
{
  "_meta": {"name": "cpg_deadoil_brugge", ...},
  "config":              { ... },
  "property_regions":    [ ... ],
  "plugin_registry": {
    "entries": [{
      "type_id":     "pc/ModelProperties@v1",
      "kind":        "pc",
      "constructor": "../../code/dead_oil_model_properties.py:ModelProperties",
      "config_model":"darts.api.type_registry:PropertyContainerConfig",
      "customizable": true,
      "doc":         "Bundled dead-oil ModelProperties property container"
    }]
  }
}
```

At load time, the loader rewrites the relative constructor path
against the preset file's directory, so the in-memory `Preset` object
carries an absolute path that survives detachment from the filesystem
— the preset still works after being passed across processes (e.g.
between the MCP server and the simulation subprocess).

---

## 10. DataRef — referencing external data

`DataRef` is the JSON-transport mechanism for "this value lives outside
the JSON file". The schema accepts three values for `kind`
(`"path" | "uri" | "object"`) plus an optional `format` hint
(`"json" | "text" | "binary"`); the resolver lives in
`darts/api/data_refs.py`.

| Kind | Example | Resolution |
|---|---|---|
| `"path"` | `{"kind": "path", "value": "perm_cube.json"}` | local file read relative to the ModelSpec file; format chosen by the optional `format` field (default JSON, plus `"text"` / `"binary"`) |
| `"uri"` | `{"kind": "uri", "value": "file:///abs/path/perm.json"}` | `file://` resolves to a local read; `object://<key>` looks up the in-memory store; other schemes raise `ValueError` |
| `"object"` | `{"kind": "object", "value": "perm_cube_key"}` | in-memory lookup against `darts.api.data_refs._OBJECT_STORE`, populated by `darts.api.data_refs.register_object(key, value)` |

The reservoir section is the main consumer:

```jsonc
"reservoir": {
  "type": "structured",
  "nx": 100, "ny": 100, "nz": 1,
  "permx": {"kind": "path", "value": "permx.json"},
  "poro":  {"kind": "object", "value": "porosity_cube"}
}
```

`StrictReservoirSpec` widens its property fields to
`ReservoirValue = ScalarOrArray | DataRef`; the builder resolves each
DataRef to a plain `float | list[float]` before constructing the
native `StructReservoirConfig`.

> **Field constraints and the DataRef union.** Per-cell reservoir fields
> deliberately carry no Pydantic numeric bounds (`gt=0`, `ge=0`, …) on
> `StrictReservoirSpec` / `StrictReservoirLayerSpec`.  Constraints would be
> applied member-by-member across the `ReservoirValue | DataRef` union and
> raise `TypeError: Unable to apply constraint ...` against the DataRef
> branch.  Validation of resolved numeric values runs in the builder
> after DataRef resolution, against the native config.

---

## 11. Auto-generating JSON from Python (`autospec`)

The inverse direction. For each existing Python model, the
`darts.api.autospec` module monkey-patches DARTS constructors so that
a model that runs

```python
m = MyModel()
m.set_reservoir(...) ; m.set_physics(...) ; ...
```

builds up a `ModelSpec` in parallel. After `m.init()`, you can dump
the spec to JSON:

```python
from darts.api.autospec import emit_json
emit_json(m, "my_model.json")
```

The output should round-trip — i.e. `darts --json my_model.json`
reproduces the model. There is a known TODO from MR !249 to add
round-trip regression tests against every reference model.

---

## 12. Migration: `model_type` → `preset`

The legacy `model_type` mechanism (closed enum of seven full-model
template IDs, each backed by a JSON under
`OPEN_DARTS_MODELS_ROOT/<id>/<id>.json`) is deprecated. Side-by-side:

| Path | Status |
|---|---|
| `set_physics({"model_type": "2ph_comp"})` | deprecated — emits `DeprecationWarning`; Pydantic surfaces a `deprecated=...` annotation; still functional |
| `set_physics({"preset": "physics/compositional/co2_brine_isothermal"})` | recommended; the preset is self-contained |
| `set_physics({"preset": "physics/dead_oil/cpg_deadoil_brugge"})` | recommended; the preset ships its own code plugin |

The migration guide lives at
[`mcp-server-langchain/docs/migration_model_type_to_preset.md`](https://gitlab.com/open-darts/mcp-server-langchain/-/blob/main/docs/migration_model_type_to_preset.md).
Removal of the `ModelType` enum is gated on (a) parity presets for all
seven IDs, and (b) re-issuing the seven per-model walkthrough prompt
texts.

---

## 13. Verification

* `pytest tests/python/test_presets.py tests/python/test_primitives.py`
  → 71 unit tests pass (preset envelope, registry lookups, `$preset`
  composition + cycle detection, every shipped preset validates
  against its Config class).
* `cd models && python json_test_suite.py` → 7/7 reference models
  match their PKLs (`2ph_comp`, `2ph_comp_solid`, `2ph_do`,
  `2ph_do_thermal`, `3ph_bo`, `3ph_comp_w`, `3ph_do`).
* `ruff check darts/api/ darts/models/darts_model.py
  darts/physics/properties/black_oil.py darts/physics/properties/iapws/`
  → clean.

---

## Status and remaining work

**Landed (this is what MR !300 + the preset programme ships):**

* `darts.api` JSON layer: schema, builder, plugin registry, DataRef,
  autospec, CLI entry-point (`darts --json …`).
* `darts.api.presets` registry: envelope, `$preset` composition,
  cycle detection, directory bindings, `DARTS_PRESET_ROOT` override,
  package-data shipping (`importlib.resources`).
* 37 shipped presets covering every evaluator family, three property
  containers, four physics presets (two self-contained), three
  reservoirs, three sim-params sets.
* Plugin-registry support on the preset envelope — the first code
  plugin (`dead_oil_model_properties.py`) ships inside the package.
* `model_type` deprecated with runtime `DeprecationWarning` and
  Pydantic `deprecated` annotation.

**Remaining (tracked in MR !300 + this thread):**

1. **Parity presets for the four remaining legacy `model_type` IDs**
   — `2ph_do`, `2ph_do_thermal`, `3ph_do`, `3ph_comp_w`. Each must
   PKL-match its reference. Then collapse the legacy seven and remove
   the enum.
2. **Convert the reference JSONs** under `models/<id>/<id>.json` to
   use `{"preset": "..."}` references against the parity presets,
   shrinking them from ~100 lines to ~20.
3. **`Strict` ↔ `Patch` factory** — replace explicit `Patch*` class
   definitions with a `make_patch_model(StrictX)` factory or paired
   tests (niketagrawal1's review request on MR !249).
4. **`autospec` round-trip tests** against every reference model
   (av-novikov's TODO on MR !249).
5. **MCP server prompts** — the seven `RUN_<ID>_PROMPT_TEXT` strings
   in `mcp-server-langchain/server/subservers/prompts.py` hard-code
   `model_type` in step 3; regenerate against parity presets.
6. **`PatchModelSpec.reservoir.layers` field type** — currently
   `list[StructReservoirConfig.ReservoirLayerConfig]`; consider
   widening to accept the same dict shape `StrictModelSpec` accepts
   for symmetric merge-patch behaviour.

The end-state is: every reference model is a few-line JSON composed
from preset references; new physics setups can be expressed without
writing Python; the MCP server publishes a dynamic preset catalog;
and the `darts/api/` Python module is a thin JSON transport over the
native `Config + from_config()` contract that every DARTS class
already implements.
