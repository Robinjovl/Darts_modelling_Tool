"""Restart round-trip sanity check for hysteresis.

Runs the smoke model for 25 days with hysteresis on, saves reservoir H5,
then builds a fresh model and calls ``load_restart_data``. Asserts
``sg_max`` on the restarted engine matches the pre-save array elementwise.
"""
import os
import shutil
import sys
import numpy as np

sys.path.insert(0, ".")
from main import CaseConfig, build_model
from darts.engines import redirect_darts_output

OUT = "restart_hys_out"
if os.path.exists(OUT):
    shutil.rmtree(OUT)

cfg = CaseConfig(
    hysteresis=True, nx=20, n_points=32, total_days=50.0,
    report_step_days=25.0, first_ts=1e-3, max_ts=5.0, mult_ts=1.5,
    stop_injection_after_days=None, start_injection_h2o_days=None,
    output_folder=OUT, write_vtk=False,
)

redirect_darts_output("restart_hys.log")

# --- phase 1: simulate + save ---
m = build_model(cfg)
m.run(25.0)
m.output.save_data_to_h5(kind="reservoir")
pre_sg_max = np.asarray(
    m.physics.get_engine_history_array("sg_max", n_blocks=m.reservoir.mesh.n_blocks),
    dtype=float,
).copy()
pre_X = np.asarray(m.physics.engine.X, copy=True)
print(f"pre-save  sg_max: nonzero={np.count_nonzero(pre_sg_max)}/{pre_sg_max.size}  max={pre_sg_max.max():.3e}")
# Stash the snapshot before phase-2 rebuild overwrites the H5 via set_output/save_initial.
stash_path = "restart_hys_snapshot.h5"
shutil.copyfile(os.path.join(OUT, "reservoir_solution.h5"), stash_path)
del m

# --- phase 2: fresh model, load from stashed snapshot ---
m2 = build_model(cfg)
m2.load_restart_data(stash_path)
post_sg_max = np.asarray(
    m2.physics.get_engine_history_array("sg_max", n_blocks=m2.reservoir.mesh.n_blocks),
    dtype=float,
).copy()
post_X = np.asarray(m2.physics.engine.X, copy=True)
print(f"post-load sg_max: nonzero={np.count_nonzero(post_sg_max)}/{post_sg_max.size}  max={post_sg_max.max():.3e}")

# reservoir-only slice (load_restart_data writes n_res_blocks rows)
n_res = m2.reservoir.mesh.n_res_blocks
ok_hist = np.allclose(pre_sg_max[:n_res], post_sg_max[:n_res], rtol=1e-10, atol=1e-14)
ok_primary = np.allclose(pre_X[: n_res * 2], post_X[: n_res * 2], rtol=1e-10, atol=1e-14)
print(f"sg_max restored:  {ok_hist}")
print(f"primary restored: {ok_primary}")
if not (ok_hist and ok_primary):
    sys.exit(1)
print("RESTART ROUND-TRIP OK")
