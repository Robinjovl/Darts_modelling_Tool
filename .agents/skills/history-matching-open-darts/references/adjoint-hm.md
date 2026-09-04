# Adjoint-gradient history matching

Status: the substrate has no adjoint driver yet. The assertive check
`workflows/evals/spikes/brugge_adjoint_check.py` (result JSON beside it) demonstrates on the
Brugge proxy: objective at truth 1.4e-27, adjoint vs central finite differences relative error
median 1.4e-5 (max 1.1e-4), angle 0.0009°.

Facts to respect (`darts/models/opt/opt_module_settings.py`):
- Controls: transmissibility (`set_res_tran`) and well index (`set_wells_tran`) only; other
  inputs need finite differences (2·D forwards, central).
- Objective: misfit of instantaneous `engine.time_data` rows at report times with production
  negated; interval-averaged report data is a different quantity.
- A forward failure inside the objective calls `sys.exit`: always run in a subprocess.
- Thermal adjoint is incomplete (warning issued); `engine_nc_nl_cpu` returns a wrong gradient.
- Gate: angle between adjoint and finite-difference gradients < 5° before optimization.

Until a driver lands, use ES-MDA; if a gradient study is still required, adapt the spike (it is
a documented, reproducible script) and journal every forward with the substrate's executor.
