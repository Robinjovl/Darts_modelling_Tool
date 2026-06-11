# T2Well Drift-Flux Verification Benchmark

Primary reference for the analytical solution and T2Well comparison:

Pan, Webb, and Oldenburg (2011), "Analytical solution for two-phase flow in a wellbore using the drift-flux model."

The current input deck runs two drift-flux verification cases:

- `20_degC_air`, with digitized analytical and T2Well profiles in `digitized_t2well_paper_profiles_20_degC_air.csv`
- `40_degC_CO2`, with digitized analytical and T2Well profiles in `digitized_t2well_paper_profiles_40_degC_CO2.csv`

Input setup used here:

- Vertical wellbore length: 1000 m
- Grid resolution: 10 m
- Inner diameter: 0.1 m
- Wall roughness: 2.4e-5 m
- Top outlet pressure: 1.0e5 Pa
- Bottom injection mass rates: 0.19625 kg/s gas and 0.19625 kg/s H2O
- Total upward mass flux: 50 kg/m2/s

By running the `main` file, generated comparison files are written to `paper_comparison/`.
