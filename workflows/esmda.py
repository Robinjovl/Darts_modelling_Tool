"""ES-MDA history matching of a per-cell log10-permeability field over ``dageo`` (design 6.3).

First implementation: diagonal observation covariance, full-cell updating with Gaspari-Cohn
localization on the distance between cell centroids and the well of each datum, bounds enforced
after every update, per-step persistence and gates. Identical-twin truth generation draws the
truth from the declared prior with a seed index disjoint from the member indices.
"""

from __future__ import annotations

import json

import numpy as np

from workflows import gates
from workflows.adapter import load_adapter
from workflows.executor import IsolatedExecutor, MemberTask, run_members
from workflows.journal import StudyStore, atomic_write_json, provenance
from workflows.members import LogPermField, Parameterization
from workflows.spec import ObservationSpec, StudySpec

TRUTH_SEED_INDEX = 999  # geology substream index reserved for the hidden truth


def gaspari_cohn(distance: np.ndarray, length: float) -> np.ndarray:
    """Fifth-order compact taper (Gaspari and Cohn 1999); zero beyond ``2 * length``."""
    r = np.asarray(distance, dtype=float) / float(length)
    taper = np.zeros_like(r)
    near = r <= 1.0
    far = (r > 1.0) & (r <= 2.0)
    x = r[near]
    taper[near] = -0.25 * x**5 + 0.5 * x**4 + 0.625 * x**3 - 5.0 / 3.0 * x**2 + 1.0
    x = r[far]
    taper[far] = (
        x**5 / 12.0
        - 0.5 * x**4
        + 0.625 * x**3
        + 5.0 / 3.0 * x**2
        - 5.0 * x
        + 4.0
        - 2.0 / (3.0 * x)
    )
    return np.clip(taper, 0.0, 1.0)


class ObservationLayout:
    """Fixed order of the observation vector: (well, quantity, report time)."""

    def __init__(self, observation: ObservationSpec):
        self.observation = observation
        times = list(observation.report_times)
        n_train = max(1, int(round(len(times) * (1.0 - observation.held_out_fraction))))
        self.entries = [
            (w, q, k)
            for w in observation.wells
            for q in observation.quantities
            for k in range(len(times))
        ]
        self.train = np.array([k < n_train for (_, _, k) in self.entries])
        self.times = times

    def vector(self, observed: dict) -> np.ndarray:
        return np.array(
            [observed["values"][f"{w}:{q}"][k] for (w, q, k) in self.entries],
            dtype=float,
        )

    def positions(self, geometry: dict) -> np.ndarray:
        return np.array(
            [geometry["wells"][w]["xyz"][:2] for (w, _, _) in self.entries], dtype=float
        )

    def sigma(self, clean: np.ndarray) -> np.ndarray:
        """Per-entry noise ``max(sigma_abs, sigma_rel * |d|, sigma_floor)`` (documented convention)."""
        obs = self.observation
        sigma = np.full(clean.shape, obs.sigma_abs or 0.0, dtype=float)
        if obs.sigma_rel:
            sigma = np.maximum(sigma, obs.sigma_rel * np.abs(clean))
        return np.maximum(sigma, obs.sigma_floor)


def _field(spec: StudySpec, geometry: dict) -> LogPermField:
    parameterization = Parameterization.from_specs(spec.parameters, geometry)
    fields = [f for f in parameterization.families if isinstance(f, LogPermField)]
    if len(fields) != 1 or len(parameterization.families) != 1:
        raise ValueError("hm-esmda expects exactly one LogPermField parameter family")
    return fields[0]


def _prepare(spec: StudySpec, study_root):
    store = StudyStore(study_root)
    store.write_spec(spec)
    adapter = load_adapter(spec.model.adapter, spec.model.model_dir)
    snapshot = adapter.snapshot(store.root / "inputs")
    geometry = adapter.geometry(snapshot)
    prov = provenance()
    prov["input_hashes"] = snapshot.files
    identities = (
        spec.identities(prov)
        if prov["engine_fingerprint"]
        else {"workflow_run_hash": spec.workflow_run_hash()}
    )
    return store, adapter, snapshot, geometry, prov, identities


def _forward_factory(
    spec, store, snapshot, identities, prov, layout, executor, prefix="w", resume=True
):
    counter = {"wave": 0}

    def forward(params: np.ndarray) -> np.ndarray:
        wave = counter["wave"]
        counter["wave"] += 1
        tasks = [
            MemberTask(
                member=f"{prefix}{wave:02d}m{i:04d}",
                study_root=str(store.root),
                adapter=spec.model.adapter,
                model_dir=spec.model.model_dir,
                snapshot={"root": snapshot.root, "files": snapshot.files},
                realization={"permx": (10.0 ** np.asarray(row, dtype=float)).tolist()},
                report_times=list(spec.observations.report_times),
                observation=spec.observations.__dict__,
                spec_hash=identities["workflow_run_hash"],
                input_hash=snapshot.hash,
                binary_fingerprint=prov.get("engine_fingerprint") or "unknown",
            )
            for i, row in enumerate(params)
        ]
        results = run_members(store, tasks, executor, resume=resume)
        failed = [
            t.member for t, r in zip(tasks, results, strict=False) if r.status != "ok"
        ]
        if failed:
            raise RuntimeError(
                f"ES-MDA wave {wave}: members failed permanently: {failed}"
            )
        return np.array([layout.vector(r.payload["observation"]) for r in results])

    return forward


def make_truth(spec: StudySpec, study_root, executor=None) -> dict:
    """Identical twin: truth from the declared prior (seed index disjoint from members), noisy data."""
    store, adapter, snapshot, geometry, prov, identities = _prepare(spec, study_root)
    field = _field(spec, geometry)
    layout = ObservationLayout(spec.observations)
    truth_index = int(spec.reference.get("truth_seed_index", TRUTH_SEED_INDEX))
    x_true = field.sample_log10(spec.seeds().generator("geology", truth_index), 1)[0]
    executor = executor or _executor(spec)
    forward = _forward_factory(
        spec, store, snapshot, identities, prov, layout, executor, prefix="t"
    )
    clean = forward(x_true[None, :])[0]
    sigma = layout.sigma(clean)
    noise = (
        spec.seeds()
        .generator("observation_noise", truth_index)
        .standard_normal(clean.size)
    )
    truth = {
        "truth_seed_index": truth_index,
        "x_true_log10": x_true.tolist(),
        "d_clean": clean.tolist(),
        "d_obs": (clean + sigma * noise).tolist(),
        "sigma": sigma.tolist(),
        "train_mask": layout.train.tolist(),
        "entries": [list(e) for e in layout.entries],
    }
    truth_dir = store.root / "truth"
    truth_dir.mkdir(exist_ok=True)
    atomic_write_json(truth_dir / "truth.json", truth)
    store.write_manifest(
        {
            "identities": identities,
            "provenance": prov,
            "seeds": spec.seeds().to_dict(),
            "stage": "truth",
        }
    )
    store.append_event(
        {"stage": "truth", "nd": int(clean.size), "n_train": int(layout.train.sum())}
    )
    return truth


def _executor(spec: StudySpec):
    return IsolatedExecutor(
        n_workers=spec.compute.max_workers,
        threads_per_member=spec.compute.threads_per_member,
        timeout_s=spec.compute.walltime_s,
        retries=spec.compute.retries,
        memory_per_member_gb=spec.compute.memory_per_member_gb,
    )


def run_esmda(spec: StudySpec, study_root, executor=None, resume: bool = True) -> dict:
    """Assimilate the training data with ES-MDA; gates and persistence after every step."""
    import dageo

    store, adapter, snapshot, geometry, prov, identities = _prepare(spec, study_root)
    truth_path = store.root / "truth" / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(
            "run `truth` first (identical twin) or provide truth/truth.json"
        )
    with open(truth_path, encoding="utf-8") as handle:
        truth = json.load(handle)
    field = _field(spec, geometry)
    layout = ObservationLayout(spec.observations)
    d = spec.design
    ne, n_steps = int(d["ne"]), int(d["n_steps"])
    alphas = d.get("alphas") or n_steps
    bounds_sigma = float(d.get("bounds_sigma", 4.0))
    lo, hi = (
        field.mean_log10 - bounds_sigma * field.sigma_log10,
        field.mean_log10 + bounds_sigma * field.sigma_log10,
    )
    train = np.array(truth["train_mask"], dtype=bool)
    d_obs, sigma = np.array(truth["d_obs"]), np.array(truth["sigma"])
    prior = field.sample_log10(spec.seeds().generator("geology"), ne)
    executor = executor or _executor(spec)
    forward_all = _forward_factory(
        spec, store, snapshot, identities, prov, layout, executor, resume=resume
    )

    history = {"data": [], "params": []}

    def forward(params):
        full = forward_all(params)
        history["data"].append(full)
        return full[:, train]

    localization = None
    length = d.get("localization_length_m")
    if length:
        centroids = np.asarray(field.centroids, dtype=float)[:, :2]
        positions = layout.positions(geometry)[train]
        dist = np.sqrt(
            ((centroids[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
        )
        localization = gaspari_cohn(dist, float(length))

    def clip(models):
        np.clip(models, lo, hi, out=models)
        history["params"].append(models.copy())

    prior_data_full = forward(prior)
    # dageo evaluates the prior data once (supplied here) and the forward after every step but the
    # last; the posterior prediction is one explicit wave below, giving (n_steps + 1) x ne runs.
    posterior = dageo.esmda(
        model_prior=prior,
        forward=forward,
        data_obs=d_obs[train],
        sigma=sigma[train],
        alphas=alphas,
        data_prior=prior_data_full,
        localization_matrix=localization,
        callback_post=clip,
        return_post_data=False,
        return_steps=False,
        random=spec.seeds().generator("assimilation"),
    )
    posterior_data = forward_all(posterior)
    history["data"].append(posterior_data)
    steps = []
    params_by_step = [prior] + history["params"]
    for k, data in enumerate(history["data"]):
        params = params_by_step[min(k, len(params_by_step) - 1)]
        steps.append(
            {
                "step": k,
                "chi2_train": gates.chi2(data[:, train], d_obs[train], sigma[train]),
                "held_out_rmse": gates.held_out_rmse(
                    data[:, ~train], d_obs[~train], sigma[~train]
                )
                if (~train).any()
                else None,
                "coverage80_held_out": gates.coverage(
                    data[:, ~train], d_obs[~train], 0.8
                )
                if (~train).any()
                else None,
                "spread_ratio": gates.spread_ratio(params, prior),
                "rmse_log10_vs_truth": float(
                    np.sqrt(
                        np.mean(
                            (params.mean(axis=0) - np.array(truth["x_true_log10"])) ** 2
                        )
                    )
                ),
            }
        )
        np.save(store.root / f"params_step{k}.npy", params)
        np.save(store.root / f"data_step{k}.npy", data)
        store.append_event({"stage": "esmda-step", **steps[-1]})
    summary = {
        "ne": ne,
        "n_steps": n_steps,
        "nd_train": int(train.sum()),
        "nd_held_out": int((~train).sum()),
        "localization_length_m": length,
        "gates_version": gates.GATES_VERSION,
        "steps": steps,
        "final_chi2_in_band": gates.chi2_band(steps[-1]["chi2_train"]),
    }
    atomic_write_json(store.root / "esmda_summary.json", summary)
    store.write_manifest(
        {
            "identities": identities,
            "provenance": prov,
            "seeds": spec.seeds().to_dict(),
            "stage": "esmda",
            "summary": summary,
        }
    )
    return summary
