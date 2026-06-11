# T2Well Drift-Flux Verification Benchmark

Primary reference for the analytical solution and T2Well comparison:

Pan, Webb, and Oldenburg (2011), "Analytical solution for two-phase flow in a wellbore using the drift-flux model."

The current input deck uses the CO2/water T2Well-ECO2N verification variant of this drift-flux benchmark family, with digitized analytical and T2Well profiles stored in `digitized_t2well_paper_profiles.csv`.

Input setup used here:

- Vertical wellbore length: 1000 m
- Grid resolution: 10 m
- Inner diameter: 0.1 m
- Wall roughness: 2.4e-5 m
- Isothermal temperature: 40 C
- Top outlet pressure: 1.0e5 Pa
- Bottom injection mass rates: 0.19625 kg/s CO2 and 0.19625 kg/s H2O
- Total upward mass flux: 50 kg/m2/s
- Simulation endpoint: 0.456869e9 s

By running the `main` file, generated comparison files are written to `paper_comparison/`.
