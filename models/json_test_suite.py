import os
import subprocess
import time

from for_each_model import _ensure_parent_dir


JSON_MODELS = [
    os.path.join("2ph_comp", "2ph_comp.json"),
    os.path.join("2ph_comp_solid", "2ph_comp_solid.json"),
    os.path.join("2ph_do", "2ph_do.json"),
    os.path.join("2ph_do_thermal", "2ph_do_thermal.json"),
    os.path.join("3ph_bo", "3ph_bo.json"),
    os.path.join("3ph_comp_w", "3ph_comp_w.json"),
    os.path.join("3ph_do", "3ph_do.json"),
]


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


def run_json_tests(base_dir=None, days=None, timeout=None):
    if os.getenv("SKIP_JSON_TESTS") == "1":
        return 0, []

    models_dir = base_dir or os.path.abspath(os.path.dirname(__file__))
    logs_dir = os.path.join(models_dir, "_logs")
    os.makedirs(logs_dir, exist_ok=True)

    days_env = _parse_optional_float(os.getenv("JSON_TEST_DAYS"))
    if days is None:
        days = 1.0 if days_env is None else days_env
    timeout_env = _parse_optional_int(os.getenv("JSON_TEST_TIMEOUT"), default=7200)
    if timeout is None:
        timeout = timeout_env

    failed = []
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

        cmd = ["darts", "--json", json_path]
        if days is not None:
            cmd += ["--days", str(days)]

        start = time.time()
        ok = False
        try:
            with open(stdout_path, "w") as stdout_file, open(
                stderr_path, "w"
            ) as stderr_file:
                result = subprocess.run(
                    cmd, stdout=stdout_file, stderr=stderr_file, timeout=timeout
                )
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            with open(stderr_path, "a") as stderr_file:
                stderr_file.write(
                    f"Timeout after {timeout} seconds running: {' '.join(cmd)}\n"
                )
            ok = False

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


if __name__ == "__main__":
    n_total, failed = run_json_tests()
    if failed:
        print("Failed JSON models:\n\t" + "\n\t".join(failed))
    exit(len(failed))
