import sys, numpy as np
sys.path.insert(0, '.')
from main import CaseConfig, build_model

cfg = CaseConfig(hysteresis=True, nx=20, n_points=32, total_days=50.0,
                 report_step_days=25.0, first_ts=1e-3, max_ts=5.0, mult_ts=1.5,
                 stop_injection_after_days=None, start_injection_h2o_days=None,
                 output_folder='hys_true_out', write_vtk=False)

from darts.engines import redirect_darts_output
redirect_darts_output('hys_true.log')

m = build_model(cfg)
print('model built OK', flush=True)
m.run(25.0)
print('step 1 done', flush=True)
m.run(25.0)
print('step 2 done', flush=True)
X = np.asarray(m.physics.engine.X, copy=True)
print('len(X)=', len(X), flush=True)
print('X[:12]=', X[:12], flush=True)
print('X stats: min=', X.min(), 'max=', X.max(), 'mean=', X.mean(), flush=True)
Xhis = np.asarray(m.physics.engine.Xhis, copy=True)
print('Xhis len=', len(Xhis), flush=True)
print('Xhis per cell (n_his=1):', Xhis[:25], flush=True)
print('Xhis stats: min=', Xhis.min(), 'max=', Xhis.max(), 'mean=', Xhis.mean(), flush=True)
# Get sg_max per reservoir cell via helper
sg_max = m.physics.get_engine_history_array('sg_max', n_blocks=m.reservoir.mesh.n_blocks)
print('sg_max per cell[:25]=', sg_max[:25], flush=True)
print('sg_max stats: min=', sg_max.min(), 'max=', sg_max.max(), 'mean=', sg_max.mean(), flush=True)
print('OK', flush=True)
