# Observations, identifiability, gates

Observations (`observations` in the spec): `wells`, `quantities` (`oil_rate`, `wat_rate`,
`gas_rate`, `bhp`, `bht`), `report_times` in days. Values are taken from the adapter's
`observe` at the report times (instantaneous rows). Production rates are positive.
Noise: `sigma = max(sigma_abs, sigma_rel·|d|, sigma_floor)` per entry; declare it once.

Identifiability:
- BHP-controlled wells observe rates, rate-controlled wells observe BHP; do not "observe" the control.
- Far-field cells barely affect the data: expect the posterior to stay near the prior there
  (spread ratio near 1 far from wells is correct, not a failure).
- A perfect training fit with a poor held-out fit is overfitting; increase localization or noise.

Gates (`workflows/gates.py`, reported per step in `esmda_summary.json`):

| Gate | Pass |
|---|---|
| chi² of normalized residuals on training data | within [0.5, 2] |
| held-out RMSE (normalized) | decreasing over steps and ≤ 2 at the end |
| coverage of held-out data by the 80 % ensemble band | ≥ 0.6 |
| spread ratio (posterior/prior std) | ≥ 0.1 (collapse below) |

Report all four; a match that fails coverage or spread is not accepted.
