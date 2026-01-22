"""
Reproduce blood glucose (without noise) and IOB from CGM attack log data.
Uses insulin input from log files to drive the virtual patient simulation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simglucose.patient.t1dm_patient import T1DMPatient, Action

CURRENT_FOLDER = Path(__file__).parent
DATA_FOLDER = CURRENT_FOLDER / "data"
OUTPUT_FOLDER = CURRENT_FOLDER / "output"


def load_adult_007_data(file_path: Path) -> pd.DataFrame:
    """Load data from adult_007 format Excel file."""
    df = pd.read_excel(file_path)
    return df[["Time (min)", "CGM Reading", "Virtual Patient Glucose", "insulin"]].copy()


def load_attacks_data(file_path: Path) -> pd.DataFrame:
    """Load data from attacks-analyzed format Excel file."""
    df = pd.read_excel(file_path, header=None)
    # Extract data starting from row 4 (0-indexed), columns 2-7
    data = df.iloc[4:, 2:8].copy()
    data.columns = [
        "timestamp",
        "Time (min)",
        "CGM_reading",
        "Patient_glucose",
        "MARD",
        "insulin",
    ]
    data = data.reset_index(drop=True)
    # Convert numeric columns
    for col in ["Time (min)", "CGM_reading", "Patient_glucose", "MARD", "insulin"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def simulate_patient(
    insulin_series: np.ndarray,
    patient_name: str = "adult#007",
    init_bg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate virtual patient with given insulin inputs.

    Args:
        insulin_series: Array of insulin values (U/min) for each time step
        patient_name: Name of patient to simulate
        init_bg: Initial blood glucose (mg/dL), if None uses patient default

    Returns:
        Tuple of (time, blood_glucose, iob) arrays
    """
    patient = T1DMPatient.withName(patient_name, init_bg=init_bg)

    time_arr = []
    bg_arr = []
    iob_arr = []

    for insulin in insulin_series:
        time_arr.append(patient.t_elapsed)
        bg_arr.append(patient.observation.Gsub)
        iob_arr.append(patient.get_iob(include_plasma=True, subtract_baseline=False))

        action = Action(CHO=0, insulin=insulin)
        patient.step(action)

    return np.array(time_arr), np.array(bg_arr), np.array(iob_arr)


def process_file(
    file_path: Path,
    patient_name: str = "adult#007",
    output_suffix: str = "_reproduced",
) -> pd.DataFrame:
    """
    Process a single data file and generate BG and IOB.

    Args:
        file_path: Path to input Excel file
        patient_name: Name of patient to simulate
        output_suffix: Suffix for output file name

    Returns:
        DataFrame with results
    """
    file_name = file_path.stem

    # Determine file format and load
    if "attacks" in file_name.lower():
        data = load_attacks_data(file_path)
        original_bg_col = "Patient_glucose"
    else:
        data = load_adult_007_data(file_path)
        original_bg_col = "Virtual Patient Glucose"

    # Get insulin series
    insulin_series = np.asarray(data["insulin"].values)

    # Get initial BG from first row if available
    init_bg: float | None = None
    if original_bg_col in data.columns:
        init_bg = float(data[original_bg_col].iloc[0])

    # Run simulation
    _, bg_arr, iob_arr = simulate_patient(
        insulin_series=insulin_series,
        patient_name=patient_name,
        init_bg=init_bg,
    )

    # Build result DataFrame
    result = data.copy()
    result["Simulated_BG"] = bg_arr
    result["IOB"] = iob_arr

    # Save output
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / f"{file_name}{output_suffix}.xlsx"
    result.to_excel(output_path, index=False)
    print(f"Saved results to: {output_path}")

    return result


def plot_results(result: pd.DataFrame, title: str, save_path: Path | None = None):
    """Plot simulated BG and IOB."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    time = result["Time (min)"].values

    # Blood Glucose
    axes[0].plot(time, result["Simulated_BG"], label="Simulated BG", color="blue")
    axes[0].axhline(y=70, color="red", linestyle="--", alpha=0.7, label="Hypo (70)")
    axes[0].axhline(y=180, color="orange", linestyle="--", alpha=0.7, label="Hyper (180)")
    axes[0].set_ylabel("Blood Glucose (mg/dL)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # IOB
    axes[1].plot(time, result["IOB"], label="IOB", color="green")
    axes[1].set_ylabel("IOB (U)")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")

    plt.show()


def main():
    PLOT_FOLDER = OUTPUT_FOLDER / "plots"
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Process adult_007 file
    adult_007_path = DATA_FOLDER / "adult_007_20251015_152715.xlsx"
    if adult_007_path.exists():
        print(f"\nProcessing: {adult_007_path.name}")
        result1 = process_file(adult_007_path, patient_name="adult#007")
        print(f"  Rows: {len(result1)}")
        print(f"  BG range: {result1['Simulated_BG'].min():.1f} - {result1['Simulated_BG'].max():.1f}")
        print(f"  IOB range: {result1['IOB'].min():.3f} - {result1['IOB'].max():.3f}")
        plot_results(
            result1,
            title="adult_007 - Simulated BG and IOB",
            save_path=PLOT_FOLDER / "adult_007_bg_iob.png",
        )

    # Process attacks-analyzed file
    attacks_path = DATA_FOLDER / "attacks-analyzed.xlsx"
    if attacks_path.exists():
        print(f"\nProcessing: {attacks_path.name}")
        result2 = process_file(attacks_path, patient_name="adult#007")
        print(f"  Rows: {len(result2)}")
        print(f"  BG range: {result2['Simulated_BG'].min():.1f} - {result2['Simulated_BG'].max():.1f}")
        print(f"  IOB range: {result2['IOB'].min():.3f} - {result2['IOB'].max():.3f}")
        plot_results(
            result2,
            title="attacks-analyzed - Simulated BG and IOB",
            save_path=PLOT_FOLDER / "attacks_analyzed_bg_iob.png",
        )


if __name__ == "__main__":
    main()
