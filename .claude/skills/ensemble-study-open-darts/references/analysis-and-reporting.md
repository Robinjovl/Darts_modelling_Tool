# Analysis and reporting

`analyze` writes `analysis.json`: per quantity (well, quantity, report time or cumulative)
percentiles P10/P50/P90 with bootstrap 90 % intervals, mean, std, failure counts; Morris
(μ*, σ) or Sobol (S1, ST, with bootstrap CI) when the design supports it.

Report:
- The design, `n`, seed, and how many members failed and why.
- Percentiles with their intervals; call an interval-wide difference "not resolved".
- Sensitivity ranking with the index and its interval; say which inputs are inert.
- What was held fixed (controls, physics) and the scope of validity of the ranges.
- Cost: simulations, wall time, member seconds, workers (from the journal).

Do not: mix studies with different `n` or ranges; report P10/P90 without intervals; call a
one-at-a-time sweep a sensitivity index.
