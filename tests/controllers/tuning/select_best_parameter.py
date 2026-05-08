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
OUTPUT_DIR = "/home/kexin/simglucose/simglucose/tests/controllers/tuning/result"

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
OUTCOME_COLS = ["low", "target", "high", "very_high"]

# =============================================================================
# Helpers
# =============================================================================


def _resolve_targets(patient_group):
    """Return (mean, sd) arrays aligned with OUTCOME_COLS for *patient_group*."""
    group_key = patient_group.split("_")[0]  # "child_002" → "child"
    targets = CLINICAL_TARGETS.get(group_key)
    if targets is None:
        raise KeyError(
            f"Unknown patient group '{patient_group}' "
            f"(derived key '{group_key}'). Known keys: {list(CLINICAL_TARGETS)}."
        )
    means = np.array([targets[m][0] for m in OUTCOME_COLS])
    sds = np.array([targets[m][1] for m in OUTCOME_COLS])
    return means, sds


def load_and_filter(csv_path, patient_group):
    """Read CSV and keep only rows belonging to *patient_group*."""
    df = pd.read_csv(csv_path)
    mask = df["virtual_patient_id"].astype(str).str.contains(patient_group, na=False)
    return df[mask].copy()


def aggregate_trials(df):
    """Average outcome columns across trials for each unique parameter combo."""
    agg_df = (
        df.groupby(PARAM_COLS)
        .agg(
            **{col: (col, "mean") for col in OUTCOME_COLS},
            virtual_patient_id=("virtual_patient_id", "first"),
        )
        .reset_index()
    )
    # Flatten MultiIndex columns from the agg dict
    agg_df.columns = [c[0] if isinstance(c, tuple) else c for c in agg_df.columns]
    return agg_df


def score_parameters(df, patient_group):
    """Add a 'score' column: z-score distance from clinical targets (lower is better)."""
    means, sds = _resolve_targets(patient_group)
    actuals = df[OUTCOME_COLS].to_numpy(dtype=float)
    z = (actuals - means) / sds
    df["score"] = np.sqrt((z**2).sum(axis=1))
    return df


def rank_and_export(df, output_path):
    """Save all parameter combos ranked by score (best first) to CSV."""
    ranked = df.sort_values("score")
    # Put virtual_patient_id first
    cols = ["virtual_patient_id"] + [c for c in ranked.columns if c != "virtual_patient_id"]
    ranked[cols].to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved {len(ranked)} ranked parameters to {output_path}")
    return ranked


def plot_results(df, patient_group, output_path):
    """2-D scatter: hyperglycemia (high+very_high) vs hypoglycemia (low), colored by deviation score."""
    group_key = patient_group.split("_")[0]
    targets = CLINICAL_TARGETS.get(group_key, CLINICAL_TARGETS["child"])
    low_mean = targets["low"][0]
    low_sd = targets["low"][1]
    hyper_mean = targets["high"][0] + targets["very_high"][0]
    hyper_sd = np.sqrt(targets["high"][1] ** 2 + targets["very_high"][1] ** 2)

    df = df.copy()
    df["hyper"] = df["high"] + df["very_high"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df["low"],
        df["hyper"],
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
        print(f"  {len(raw)} rows")

        agg = aggregate_trials(raw)
        print(f"  {len(agg)} unique parameter combinations")

        scored = score_parameters(agg, group)
        rank_and_export(scored, f"{OUTPUT_DIR}/best_parameters_{group}.csv")
        plot_results(scored, group, f"{OUTPUT_DIR}/parameter_sweep_{group}.png")


if __name__ == "__main__":
    main()
