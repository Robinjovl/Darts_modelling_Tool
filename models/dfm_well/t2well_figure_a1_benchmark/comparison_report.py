import numpy as np

from plot_paper_comparison import _final_profile, _load_reference_profiles


QUANTITIES = [
    ("pressure_bar", "Pressure [bar]"),
    ("sG", "Gas saturation [-]"),
    ("vG_m_s", "Gas phase velocity [m/s]"),
    ("drift_velocity_m_s", "Drift velocity [m/s]"),
]


def _profile_for_comparison(model):
    profile = _final_profile(model)
    profile["pressure_bar"] = profile["pressure"]
    profile["drift_velocity_m_s"] = profile["vG_m_s"] - (
        profile["sG"] * profile["vG_m_s"] + profile["sL"] * profile["vL_m_s"]
    )
    return profile


def _score_at_depths(candidate_depth, candidate_value, reference_depth, reference_value):
    predicted = np.interp(reference_depth, candidate_depth, candidate_value)
    residual = predicted - reference_value
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((reference_value - np.mean(reference_value)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return r2


def _compare_against_analytical(model, reference_profile_file):
    profile = _profile_for_comparison(model)
    reference, _ = _load_reference_profiles(reference_profile_file)
    if reference is None:
        return []

    rows = []
    for quantity, label in QUANTITIES:
        analytical = reference[
            (reference["solution"] == "Analytical")
            & (reference["quantity"] == quantity)
        ].sort_values("depth_m")
        t2well = reference[
            (reference["solution"] == "T2Well") & (reference["quantity"] == quantity)
        ].sort_values("depth_m")
        if analytical.empty or t2well.empty:
            continue

        analytical_depth = analytical["depth_m"].to_numpy()
        analytical_value = analytical["value"].to_numpy()

        common_min = max(
            float(profile["depth_m"].min()),
            float(t2well["depth_m"].min()),
            float(analytical_depth.min()),
        )
        common_max = min(
            float(profile["depth_m"].max()),
            float(t2well["depth_m"].max()),
            float(analytical_depth.max()),
        )
        common = (analytical_depth >= common_min) & (analytical_depth <= common_max)
        common_depth = analytical_depth[common]
        common_value = analytical_value[common]
        if len(common_depth) < 2:
            continue

        darts_r2 = _score_at_depths(
            profile["depth_m"].to_numpy(),
            profile[quantity].to_numpy(),
            common_depth,
            common_value,
        )
        t2well_r2 = _score_at_depths(
            t2well["depth_m"].to_numpy(),
            t2well["value"].to_numpy(),
            common_depth,
            common_value,
        )
        rows.append(
            {
                "quantity": label,
                "darts_r2": darts_r2,
                "t2well_r2": t2well_r2,
            }
        )
    return rows


def print_r2_report(model, reference_profile_file, case_label=None):
    rows = _compare_against_analytical(model, reference_profile_file)
    if not rows:
        print("\nR2 report against analytical solution was not generated.")
        return

    if case_label:
        print(f"\nCoefficient of determination against analytical solution: {case_label}")
    else:
        print("\nCoefficient of determination against analytical solution")
    print(f"{'Quantity':<28} {'DARTS-well':>12} {'T2Well':>12}")
    print(f"{'-' * 28} {'-' * 12:>12} {'-' * 12:>12}")
    for row in rows:
        print(
            f"{row['quantity']:<28} "
            f"{row['darts_r2']:>12.6f} {row['t2well_r2']:>12.6f}"
        )
