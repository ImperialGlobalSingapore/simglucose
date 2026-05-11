import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Configuration — change these for each analysis run
# =============================================================================

CSV_PATH = (
    "/home/kexin/simglucose/simglucose/tests/controllers/"
    "t1dm_patient/imgs/oref0_parameter_tuning/20260503_215120/results.csv"
)
PATIENT_GROUP = (
    "child"  # prefix filter on virtual_patient_id (e.g. "child", "adult", "adolescent")
)
OUTPUT_DIR = (
    "/home/kexin/simglucose/simglucose/tests/controllers/tuning/results/20260503_215120"
)

# Clinical TIR benchmarks: (mean%, sd%) per metric
# Source: user-provided reference table
CLINICAL_TARGETS = {
    "child": {
        "very_high": (9.3, 6.0),
        "high": (21.1, 6.8),
        "target": (67.5, 11.5),
        "low": (2.1, 1.5),
    },
    "adolescent": {  # adolescents use adult clinical targets
        "very_high": (5.6, 4.9),
        "high": (18.2, 8.4),
        "target": (74.5, 11.9),
        "low": (1.6, 2.1),
    },
    "adult": {
        "very_high": (5.6, 4.9),
        "high": (18.2, 8.4),
        "target": (74.5, 11.9),
        "low": (1.6, 2.1),
    },
}

# Columns that identify a unique parameter combination (exclude derived/redundant ones like isfProfile)
PARAM_COLS = ["sens", "dia", "carb_ratio", "min_bg", "max_bg", "max_iob", "max_basal"]
# Raw outcome columns read from the simulator CSV
RAW_OUTCOME_COLS = ["low", "target", "high", "very_high"]
# Outcome columns used for scoring: high + very_high are combined into above_tir to avoid
# the spurious-NaN problem when BG never crosses the very_high threshold.
OUTCOME_COLS = ["low", "target", "above_tir"]

# =============================================================================
# Helpers
# =============================================================================


def _resolve_targets(patient_group):
    """Return (mean, sd) arrays aligned with OUTCOME_COLS for *patient_group*."""
    group_key = patient_group.split("_")[0]  # "child_002" → "child"
    raw = CLINICAL_TARGETS.get(group_key)
    if raw is None:
        raise KeyError(
            f"Unknown patient group '{patient_group}' "
            f"(derived key '{group_key}'). Known keys: {list(CLINICAL_TARGETS)}."
        )
    # Combine high + very_high → above_tir: sum the means; combine SDs as sqrt(SD1²+SD2²)
    # under independence (covariance not reported in clinical TIR literature).
    combined = {
        "low": raw["low"],
        "target": raw["target"],
        "above_tir": (
            raw["high"][0] + raw["very_high"][0],
            np.sqrt(raw["high"][1] ** 2 + raw["very_high"][1] ** 2),
        ),
    }
    means = np.array([combined[m][0] for m in OUTCOME_COLS])
    sds = np.array([combined[m][1] for m in OUTCOME_COLS])
    return means, sds


def load_and_filter(csv_path, patient_group):
    """Read CSV and keep only rows belonging to *patient_group*."""
    df = pd.read_csv(csv_path)
    mask = df["virtual_patient_id"].astype(str).str.contains(patient_group, na=False)
    df = df[mask].copy()
    # Meal-carb scenario is encoded only in virtual_patient_id (e.g. "..._carb_56_trial_0").
    df["carb_label"] = df["virtual_patient_id"].str.extract(r"_carb_(\d+)_")
    # Patient + parameter-set identifier (e.g. "adult_007_param_575"), shared across all
    # carb/trial runs of the same controller-param combo.
    df["patient_param_id"] = df["virtual_patient_id"].str.extract(r"^([a-z]+_\d+_param_\d+)")
    # Upstream simulator drops zero-time-in-bucket values (time_in_range.py:182 filter).
    # NaN in a TIR outcome therefore means "0% time" was measured — restore the dropped zeros.
    df[RAW_OUTCOME_COLS] = df[RAW_OUTCOME_COLS].fillna(0)
    return df


def aggregate_per_scenario(df):
    """One row per (parameter combo, carb scenario): mean outcomes across trials."""
    keys = PARAM_COLS + ["carb_label"]
    agg_df = (
        df.groupby(keys)
        .agg(
            **{col: (col, "mean") for col in RAW_OUTCOME_COLS},
            patient_param_id=("patient_param_id", "first"),
        )
        .reset_index()
    )
    agg_df.columns = [c[0] if isinstance(c, tuple) else c for c in agg_df.columns]
    # Combined "above target" bucket: high + very_high. min_count=1 keeps the result
    # NaN only when BOTH parts are missing; otherwise treats the missing part as 0.
    agg_df["above_tir"] = agg_df[["high", "very_high"]].sum(axis=1, min_count=1)
    return agg_df


def score_per_scenario(df, patient_group):
    """Add per-(combo, carb) 'score': z-score distance from clinical targets (lower=better)."""
    means, sds = _resolve_targets(patient_group)
    actuals = df[OUTCOME_COLS].to_numpy(dtype=float)
    z = (actuals - means) / sds
    df["score"] = np.sqrt((z**2).sum(axis=1))
    return df


def collapse_across_scenarios(df):
    """One row per combo: mean of per-scenario scores (skipna) and mean outcomes; drops all-NaN-score combos."""
    mean_cols = ["low", "target", "above_tir", "score"]
    agg_df = (
        df.groupby(PARAM_COLS)
        .agg(
            **{col: (col, "mean") for col in mean_cols},
            patient_param_id=("patient_param_id", "first"),
        )
        .reset_index()
    )
    agg_df.columns = [c[0] if isinstance(c, tuple) else c for c in agg_df.columns]
    return agg_df.dropna(subset=["score"])


def rank_and_export(df, output_path):
    """Save all parameter combos ranked by score (best first) to CSV."""
    cols = ["patient_param_id", *PARAM_COLS, "low", "target", "above_tir", "score"]
    ranked = df.sort_values("score")[cols]
    ranked.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved {len(ranked)} ranked parameters to {output_path}")
    return ranked


def plot_results(df, patient_group, output_path):
    """2-D scatter: above-target TIR vs hypoglycemia (low), colored by deviation score."""
    means, sds = _resolve_targets(patient_group)
    low_mean, _, hyper_mean = means
    low_sd, _, hyper_sd = sds

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df["low"],
        df["above_tir"],
        c=df["score"],
        cmap="viridis_r",
        s=30,
        edgecolors="none",
        alpha=0.6,
    )
    # Low: mean ± 1 SD
    ax.axvline(
        low_mean,
        color="orange",
        linestyle="--",
        alpha=0.9,
        label=f"Clinical low mean ({low_mean:.1f}%)",
    )
    ax.axvspan(low_mean - low_sd, low_mean + low_sd, color="orange", alpha=0.1)
    # Hyper: mean ± 1 SD
    ax.axhline(
        hyper_mean,
        color="red",
        linestyle="--",
        alpha=0.9,
        label=f"Clinical high+very_high mean ({hyper_mean:.1f}%)",
    )
    ax.axhspan(hyper_mean - hyper_sd, hyper_mean + hyper_sd, color="red", alpha=0.1)

    ax.set_xlabel("% time low (<70 mg/dL)")
    ax.set_ylabel("% time high + very high (>180 mg/dL)")
    ax.set_title(
        f"Parameter sweep — {patient_group}\n"
        "Color = deviation from clinical TIR targets (lower = better)"
    )
    ax.legend(loc="upper right")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Deviation from clinical targets (z-score)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


# =============================================================================
# Main
# =============================================================================


def _discover_groups(csv_path):
    """Return sorted list of unique patient-group prefixes found in the CSV."""
    df = pd.read_csv(csv_path, usecols=["virtual_patient_id"])
    ids = df["virtual_patient_id"].astype(str)
    # Extract prefix: "child_002_param_..." → "child_002", "adult_007_param_..." → "adult_007"
    groups = ids.str.extract(r"^([a-z]+_\d+)", expand=False).dropna().unique()
    return sorted(groups)


def main():
    groups = _discover_groups(CSV_PATH)
    print(f"Found {len(groups)} patient groups: {groups}")

    for group in groups:
        print(f"\n--- {group} ---")
        raw = load_and_filter(CSV_PATH, group)
        print(f"  {len(raw)} raw rows")

        per_scenario = aggregate_per_scenario(raw)
        print(f"  {len(per_scenario)} (combo × carb scenario) rows after trial averaging")

        per_scenario_scored = score_per_scenario(per_scenario, group)
        per_combo = collapse_across_scenarios(per_scenario_scored)
        print(f"  {len(per_combo)} unique parameter combinations with valid mean score")

        rank_and_export(per_combo, f"{OUTPUT_DIR}/best_parameters_{group}.csv")
        plot_results(per_combo, group, f"{OUTPUT_DIR}/parameter_sweep_{group}.png")


if __name__ == "__main__":
    main()
