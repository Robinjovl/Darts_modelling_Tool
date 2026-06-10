# 2ph_hysteresis

This model is a small 1D two-phase CO2/H2O example used to demonstrate hysteresis support in the OBL-based compositional workflow.

It uses:
- a 1D structured mesh with `nx = 100`, `ny = 1`, `nz = 1`
- two phases: aqueous (`Aq`) and vapor/gas (`V`)
- two components: `H2O` and `CO2`
- a single injector cell and a single producer controlled by BHP

The default schedule starts with CO2 injection, stops injection after
400 days, and switches to water injection after 800 days. The case is designed
to illustrate the effect of hysteresis on relative permeability and capillary
pressure during drainage, shut-in, and subsequent imbibition/re-injection
stages.

At the framework level, hysteresis is disabled by default, so existing models
that do not declare history variables are unaffected. This particular example
turns hysteresis on intentionally in order to exercise the `sg_max` history
workflow.

## Files

- `main.py`: run script and case configuration
- `model.py`: model definition, physics setup, schedule logic, and history-field
  update
- `LookupTable.txt`: tabulated drainage and imbibition data used by the
  hysteresis property evaluators
- `ref/*.pkl`: reference regression results from CI pipelines

## What `LookupTable.txt` contains

`LookupTable.txt` stores tabulated saturation-dependent property curves used by
the hysteresis model. In particular, it contains:
- wetting-phase relative permeability data
- non-wetting relative permeability data for both drainage and imbibition
- drainage and imbibition capillary-pressure tables

The tables are keyed by wetting saturation and are read by
`KilloughRelPermTable` and `KilloughCapillaryPressureTable` in
`model.py`. They provide the input scanning/drainage/imbibition behavior that
is combined with the history variable `sg_max` to evaluate hysteretic flow
properties.

## Turning hysteresis on and off

In this example, hysteresis is controlled by the `hys` / `hysteresis` switch:

- `hysteresis=True` (or `Model(hys=True)`) enables the history variable `sg_max`
- `hysteresis=False` (or `Model(hys=False)`) runs the same case in drainage-only mode

At the framework level, hysteresis is enabled by passing non-empty
`history_fields=[HistoryField(...)]` into `Compositional(...)`. If no history
fields are declared, the model falls back to the standard primary-variable-only
OBL state.

## Purpose

This case is meant to isolate the impact of hysteresis in a minimal setting,
without the extra complexity of multidimensional geometry or multiple wells. It
is therefore useful both as a regression target and as a compact example of how
to activate history-variable-based hysteresis in open-DARTS.
