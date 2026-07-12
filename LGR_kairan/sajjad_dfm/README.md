# No-LGR DFM CO2 Injection Case

This is a first-step coupled DFM well/reservoir model for CO2 injection into an
aquifer. It intentionally excludes LGR and production wells.

The initial reservoir is water-filled and the injector is CO2-filled. The
injector bottom pressure and temperature are initialized to the same pressure
and temperature as the perforated reservoir cell: `200 bar` and `356.15 K`.
CO2 is injected into segment `0` of the DFM well. The run writes
`initial_equilibrium_check.csv` before taking any time steps.

Postprocessing writes two separate well time-series files:

- `dfm_well_time_data.csv`: DFM-specific diagnostics, including true BHP/BHT
  at the perforated segment and the imposed segment-0 source rate.
- `well_connection_time_data.csv`: generic DARTS connection-rate diagnostics
  from `store_well_time_data()`. In DFM cases, its `BHP/BHT` columns refer to
  the wellhead cell, so use them as WHP/WHT rather than true bottom-hole values.

Run from this directory with:

```powershell
conda run -n dev_env python main.py
```
