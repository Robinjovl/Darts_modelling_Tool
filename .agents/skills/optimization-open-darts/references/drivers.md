# Drivers beyond exhaustive placement — status and protocol

| Driver | Status | Protocol when implemented |
|---|---|---|
| Adjoint controls (T, WI) | spike only (see history-matching `adjoint-hm.md`) | gradient check angle < 5°, line search 10–45 forwards, journal every forward |
| BHP/rate controls | not implemented | finite differences (2·D forwards) or CMA-ES with common random numbers; `estimate` covers `fd_gradient_runs` |
| GBT surrogate (XGBoost/CatBoost) | not implemented; deps listed in `workflows/requirements.txt` | database from an ensemble study (≤ 300 points), hold-out R² reported, expected-improvement proposals, every proposal re-simulated, regret vs exhaustive on the smoke bed |
| Robust / CVaR | not implemented | objective over an ensemble with common random numbers; CVaR at 0.2 of the lower tail |
| Gas-EOR composition | not implemented | simplex transform of injected composition; gas utilization as secondary objective |
| CCS on the full-field bed | deferred with the bed | injection schedule; pressure footprint constraint |

Do not present a driver from this table as available; offer the exhaustive placement, an
ensemble-based screening, or a documented plan with its simulation count.
