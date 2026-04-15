"""JSON regression test suite for DARTS ModelSpec cases.

The default mode runs all `JSON_MODELS` entries, one case per subprocess,
and stores detailed stdout/stderr logs under `models/_logs`.
Each case is validated against a reference PKL file in the corresponding
`<model>/ref` folder using `CICDModel.check_performance`.

Typical entry point:
    darts json_test_suite.py
"""

import argparse
import json
import os
import platform
import subprocess
import time

JSON_MODELS = [
    os.path.join("2ph_comp", "2ph_comp.json"),
    os.path.join("2ph_comp_solid", "2ph_comp_solid.json"),
    os.path.join("2ph_do", "2ph_do.json"),
    os.path.join("2ph_do_thermal", "2ph_do_thermal.json"),
    os.path.join("3ph_bo", "3ph_bo.json"),
    os.path.join("3ph_comp_w", "3ph_comp_w.json"),
    os.path.join("3ph_do", "3ph_do.json"),
    # os.path.join("cpg_sloping_fault", "brugge_deadoil.json"),
]


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _parse_optional_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in ("none", "null"):
        return None
    return float(value)


def _parse_optional_int(value, default=None):
    if value is None:
        return default
    value = str(value).strip()
    if not value or value.lower() in ("none", "null"):
        return default
    return int(value)


def _get_pkl_suffix_candidates():
    if os.getenv("TEST_GPU") == "1":
        return ["_gpu"]
    odls = os.getenv("ODLS")
    if odls == "-a":
        return ["_iter"]
    if odls:
        return ["_odls"]
    return ["_odls", "_iter"]


def _get_platform():
    return "gpu" if os.getenv("TEST_GPU") == "1" else "cpu"


def _get_reference_perf_file(model_dir, pkl_suffix):
    system_prefix = platform.system().lower()[:3]
    return os.path.join(model_dir, "ref", f"perf_{system_prefix}{pkl_suffix}.pkl")


def _select_reference_perf_file(model, model_dir, overwrite):
    suffixes = _get_pkl_suffix_candidates()
    preferred_file = _get_reference_perf_file(model_dir, suffixes[0])

    if overwrite:
        return preferred_file

    existing_files = []
    for suffix in suffixes:
        perf_file = _get_reference_perf_file(model_dir, suffix)
        if os.path.exists(perf_file):
            existing_files.append(perf_file)
    if not existing_files:
        return None
    if len(existing_files) == 1:
        return existing_files[0]

    current = model.get_performance_data()
    metric_keys = ("timesteps", "newton iterations", "linear iterations")
    best_file = existing_files[0]
    best_score = None
    for perf_file in existing_files:
        reference = model.load_performance_data(perf_file)
        if not reference:
            continue
        score = 0
        for key in metric_keys:
            value = current.get(key)
            ref_value = reference.get(key)
            if isinstance(value, int) and isinstance(ref_value, int):
                score += abs(value - ref_value)
        if best_score is None or score < best_score:
            best_score = score
            best_file = perf_file
    return best_file


def _extract_reference_from_log(log_path):
    reference_file = None
    try:
        with open(log_path) as fp:
            for line in fp:
                if line.startswith("Reference: "):
                    reference_file = line.split("Reference:", 1)[1].strip()
    except OSError:
        return None
    return reference_file


def _parse_model_spec(json_path):
    from darts.api import ModelSpec

    with open(json_path) as fp:
        spec_dict = json.load(fp)

    if hasattr(ModelSpec, "model_validate"):
        return ModelSpec.model_validate(spec_dict)
    return ModelSpec.parse_obj(spec_dict)


def _configure_output(model):
    out_spec = getattr(model, "_output_spec", None)
    if out_spec is not None:
        folder = (
            out_spec.folder if getattr(out_spec, "folder", None) is not None else "output"
        )
        precision = (
            out_spec.precision
            if getattr(out_spec, "precision", None) is not None
            else "d"
        )
        model.set_output(output_folder=folder, precision=precision)
    else:
        model.set_output()


def _run_single_json_model(json_path, days=None):
    """Run one JSON model and compare the final state with its reference PKL.

    Args:
        json_path: Absolute or relative path to the JSON ModelSpec file.
        days: Optional simulation days override passed to `model.run(days=...)`.

    Returns:
        `0` if the simulation and reference comparison pass, `1` otherwise.
    """
    from darts.api import ModelBuilder
    from darts.api.json_model import JsonModel
    from darts.models.cicd_model import CICDModel

    class JsonCICDModel(JsonModel, CICDModel):
        pass

    json_path = os.path.abspath(json_path)
    model_dir = os.path.dirname(json_path)
    overwrite = 0
    if os.getenv("UPLOAD_PKL") == "1":
        print("UPLOAD_PKL=1 detected, forcing overwrite=0 for reference validation.")

    if not os.path.exists(json_path):
        print(f"JSON model file does not exist: {json_path}")
        return 1

    model_spec = _parse_model_spec(json_path)

    cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        model = JsonCICDModel()
        ModelBuilder.apply(model_spec, model, base_path=model_dir)

        run_platform = _get_platform()
        if run_platform == "gpu":
            gpu_device = os.getenv("GPU_DEVICE")
            if gpu_device is not None:
                from darts.engines import set_gpu_device

                set_gpu_device(int(gpu_device))
        model.init(platform=run_platform)

        _configure_output(model)
        if days is None:
            model.run()
        else:
            model.run(days=days)
        model.print_stat()
        model.print_timers()

        perf_file = _select_reference_perf_file(model, model_dir, overwrite=overwrite)
        if perf_file is None:
            print(f"Reference file does not exist under {os.path.join(model_dir, 'ref')}")
            return 1
        print(f"Reference: {perf_file}")

        failed = model.check_performance(
            overwrite=overwrite,
            perf_file=perf_file,
        )
        return int(failed)
    except Exception as err:
        print(err)
        return 1
    finally:
        os.chdir(cwd)


def run_json_tests(base_dir=None, days=None, timeout=None):
    """Run all configured JSON models and aggregate pass/fail results.

    For each case this function spawns a subprocess that executes this same
    script in `--single-json` mode, then verifies that a reference check was
    actually performed by parsing the case log for a `Reference: ...` marker.

    Args:
        base_dir: Models directory containing model subfolders and JSON files.
        days: Optional simulation days override applied to each case.
        timeout: Per-case subprocess timeout in seconds.

    Returns:
        Tuple `(n_total, failed)` where:
            - `n_total` is total number of configured JSON cases,
            - `failed` is a list of failed case relative paths.
    """
    if os.getenv("SKIP_JSON_TESTS") == "1":
        return 0, []

    models_dir = base_dir or os.path.abspath(os.path.dirname(__file__))
    logs_dir = os.path.join(models_dir, "_logs")
    os.makedirs(logs_dir, exist_ok=True)

    days_env = _parse_optional_float(os.getenv("JSON_TEST_DAYS"))
    if days is None:
        days = days_env
    timeout_env = _parse_optional_int(os.getenv("JSON_TEST_TIMEOUT"), default=7200)
    if timeout is None:
        timeout = timeout_env
    omp_threads_override = os.getenv("JSON_TEST_OMP_NUM_THREADS")

    failed = []
    runner_path = os.path.abspath(__file__)
    label_width = max(len(f"JSON {p}:") for p in JSON_MODELS)
    status_width = 4
    time_width = 8
    for rel_path in JSON_MODELS:
        json_path = os.path.join(models_dir, rel_path)
        safe_name = rel_path.replace(os.sep, "__")
        stdout_path = os.path.join(logs_dir, f"json_{safe_name}.log")
        stderr_path = os.path.join(logs_dir, f"json_{safe_name}_err.log")
        _ensure_parent_dir(stdout_path)
        _ensure_parent_dir(stderr_path)

        if not os.path.exists(json_path):
            failed.append(rel_path + " (missing)")
            label = f"JSON {rel_path}:"
            print(
                f"{label:<{label_width}} "
                f"{'FAIL':<{status_width}} "
                f"{'missing':>{time_width}}"
            )
            continue

        cmd = ["darts", runner_path, "--single-json", json_path]
        if days is not None:
            cmd += ["--days", str(days)]

        start = time.time()
        ok = False
        try:
            with open(stdout_path, "w") as stdout_file, open(
                stderr_path, "w"
            ) as stderr_file:
                env = os.environ.copy()
                if omp_threads_override is not None:
                    env["OMP_NUM_THREADS"] = str(
                        _parse_optional_int(omp_threads_override, 1)
                    )
                result = subprocess.run(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=timeout,
                    env=env,
                )
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            with open(stderr_path, "a") as stderr_file:
                stderr_file.write(
                    f"Timeout after {timeout} seconds running: {' '.join(cmd)}\n"
                )
            ok = False

        reference_file = _extract_reference_from_log(stdout_path)
        if reference_file is None:
            ok = False
            with open(stderr_path, "a") as stderr_file:
                stderr_file.write(
                    "Reference check marker not found. Expected `Reference: ...` line.\n"
                )

        elapsed = time.time() - start
        label = f"JSON {rel_path}:"
        status = "OK" if ok else "FAIL"
        print(
            f"{label:<{label_width}} "
            f"{status:<{status_width}} "
            f"{elapsed:>{time_width}.2f} s"
        )
        if not ok:
            failed.append(rel_path)

    return len(JSON_MODELS), failed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--days", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.single_json:
        exit(_run_single_json_model(args.single_json, days=args.days))

    n_total, failed = run_json_tests(
        base_dir=args.base_dir,
        days=args.days,
        timeout=args.timeout,
    )
    if failed:
        print("Failed JSON models:\n\t" + "\n\t".join(failed))
    exit(len(failed))
