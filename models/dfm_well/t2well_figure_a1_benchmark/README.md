# T2Well Figure A1 Benchmark

This case reproduces the Appendix A / Figure A1 wellbore-only verification from:

Pan et al. (2011), "Transient CO2 leakage and injection in wellbore-reservoir systems for geologic carbon sequestration."

Paper setup used here:

- Vertical wellbore length: 1000 m
- Grid resolution: 10 m
- Inner diameter: 0.1 m
- Wall roughness: 2.4e-5 m
- Isothermal temperature: 40 C
- Top outlet pressure: 1.0e5 Pa
- Bottom injection mass rates: 0.19625 kg/s CO2 and 0.19625 kg/s H2O
- Total upward mass flux: 50 kg/m2/s
- Paper steady-time endpoint: 0.456869e9 s

By running the `main` file, generated comparison files are written to `paper_comparison/`.
