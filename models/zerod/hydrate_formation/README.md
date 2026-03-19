# Hydrate Formation 0D Example

This example adds two `ZerodModel` reductions of the hydrate publication model from
`/oahu/data/avnovikov/darts-models/publications/26_hydrate`:

- `ch4`: methane hydrate formation based on the `Moridis_CH4form` setup
- `co2`: CO2 hydrate formation based on the `Li_CO2inj` setup

The reduction keeps the publication flash and kinetic formulations, but replaces the
1D/radial transport problem with a single lumped control volume driven by:

- hydrate kinetics from the original model
- pressure control when appropriate (`co2`)
- an external heat-exchange term that mimics the radial boundary cooling

## Run

From this folder:

```bash
PYTHONPATH=/oahu/data/avnovikov/open-darts-zerod \
conda run -n zerod python main.py --case all
```

Outputs are written into `output/ch4` and `output/co2`.

For each case:

- `darts.log` contains redirected DARTS/C++ output
- `simulation.log` contains Python stdout, including the per-timestep
  `ZerodModel.run()` lines

Use `--quiet` if you want to suppress those timestep lines.

## Notes

- The 0D runs are not intended to reproduce the full spatial distributions of the
  original reservoir/core models.
- They are intended to reproduce the dominant local thermo-kinetic response and
  trend similarly to the original publication time series.
