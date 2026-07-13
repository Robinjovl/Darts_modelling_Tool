DFM CO2 Injection-Production Scenario

This is a coupled DFM well/reservoir model for CO2 injection into an
aquifer with one DFM injector and one DFM producer.

The initial reservoir is water-filled. The injector is initially gas-filled, and
the producer is initially water-filled to avoid the liquid-CO2 region near the
producer control block. The producer is initialized from a `1 bar` WHP, and the
initial reservoir pressure is derived from the hydrostatic pressure at the
producer perforation instead of being prescribed as a fixed `p_init` default.
With the current water-filled producer column, this gives about `190.23 bar`.
The bottom temperature of each well is initialized to `356.15 K`. CO2 is injected
into segment `0` of the injector, and the producer uses the built-in DARTS DFM
WHP control directly at `1 bar`. The run writes `initial_equilibrium_check.csv`
before taking any time steps.

Postprocessing writes two separate well time-series files:

- `dfm_well_time_data.csv`: DFM-specific diagnostics, including true BHP/BHT
  at the perforated segment, the imposed injector source rate, and the producer
  WHP target.
- `well_connection_time_data.csv`: generic DARTS connection-rate diagnostics
  from `store_well_time_data()`. In DFM cases, its `BHP/BHT` columns refer to
  the wellhead cell, so use them as WHP/WHT rather than true bottom-hole values.
