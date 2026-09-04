# Method selection

| Method | Status in `workflows/` | Use when | Cost | Caveats |
|---|---|---|---|---|
| ES-MDA (dageo) | implemented (`hm-esmda`), diagonal noise covariance, full-cell log10-perm field, spatial localization | many uncertain inputs, ensemble output wanted, no gradients | `(n_steps+1)·ne` simulations | linear-Gaussian update; needs `ne ≥ 30–100` or localization; bounded by clipping |
| Adjoint gradient | verified spike only (`workflows/evals/spikes/brugge_adjoint_check.py`); no driver yet | few deterministic parameters, smooth objective | 1 forward + 1 adjoint per iteration | controls are transmissibility and well index only; other inputs need finite differences; thermal adjoint incomplete; `engine_nc_nl_cpu` gradient is wrong; the forward crash exits the process (run in a subprocess) |
| DiWA proxy | not available (MR !330 unmerged, private data) | fast approximate matching on large fields | seconds per evaluation after training | needs the fine-grid proxy inputs; not part of this repository |

Choose ES-MDA by default. Move to adjoint only for ≤ 20 parameters that map to
transmissibility or well index, after the gradient check passes (angle < 5° vs finite differences).
