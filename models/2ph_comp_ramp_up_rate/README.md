# Two-Phase Compositional Ramp-Up Rate Example

This example shows how to use the `idata.well_data` schedule to ramp a well rate
control from an initial rate to a target rate.

## What This Example Demonstrates

- Injector `I1` is controlled by `MOLAR_RATE`.
- Producer `P1` is controlled by BHP.
- The injector target rate ramps from `0` to `200 kmol/day`.
- The ramp lasts `10 days`.
- `ramp_up_steps=100` creates `101` scheduled controls, including both endpoints.
- `DartsModel.run()` stops at scheduled well-control times so ramp targets are
  not skipped by large timesteps.

## Key API

Ramp-up is defined through `idata.well_data`:

```python
self.idata.well_data.add_inj_rate_control(
    name="I1",
    rate=200.0,
    rate_type=well_control_iface.MOLAR_RATE,
    phase_name="gas",
    inj_composition=inj_composition,
    time=0.0,
    ramp_up_period=10.0,
    ramp_up_steps=100,
)
```

The same ramp arguments are available for producers through:

```python
self.idata.well_data.add_prd_rate_control(...)
```

## Ramp Formula

For each ramp step:

```text
fraction = step / ramp_up_steps
control_time = time + fraction * ramp_up_period
control_rate = ramp_up_start_rate + fraction * (rate - ramp_up_start_rate)
```

For this example:

```text
time = 0.0, 0.1, 0.2, ..., 10.0 days
rate = 0.0, 2.0, 4.0, ..., 200.0 kmol/day
```

## Important Notes

Ramp-up support is automatic only for controls defined through
`idata.well_data`. Direct calls to:

```python
self.physics.set_well_controls(...)
```

apply one immediate control and do not create intermediate ramp targets.

During initialization, this example delegates:

```python
set_well_controls() -> set_well_controls_idata(time=0.0)
```

After initialization, `DartsModel.run()` applies the later scheduled ramp points.
