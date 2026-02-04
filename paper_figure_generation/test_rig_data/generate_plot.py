"""
Generate plots from test rig CSV data.

This script loads CSV data from the data/ folder and generates plots.
Adjust layout and styling here without re-running simulations.
"""

import csv
from datetime import datetime
from pathlib import Path
from plot import plot_bg_cho_iob_and_save, plot_merged_and_save

file_path = Path(__file__).resolve()
parent_folder = file_path.parent

data_dir = parent_folder / "data"
result_dir = parent_folder / "result"
result_dir.mkdir(parents=True, exist_ok=True)

csv_groups = [
    ("adult_007_20260114_095204.csv", "adult_007_20260109_101615.csv"),
    ("adult_007_20260113_110541.csv", "adult_007_20260109_140030.csv"),
    ("adult_007_20260113_145352.csv", "adult_007_20260112_141550.csv"),
]


def load_test_rig_data(csv_file):
    """
    Load test rig data from CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        dict: Dictionary with keys 't', 'CGM', 'BG', 'IOB', 'CHO'
    """
    timestamps = []
    CGM = []
    BG = []
    IOB = []
    CHO = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamps.append(row["timestamp"])
            CGM.append(float(row["CGM_reading"]))
            BG.append(float(row["Paitent_glucose"]))
            IOB.append(float(row["patient_iob"]))
            CHO.append(float(row["CHO"]))

    # Convert timestamps to relative time in minutes
    t = []
    if timestamps:
        first_time = datetime.fromisoformat(timestamps[0])
        for ts in timestamps:
            current_time = datetime.fromisoformat(ts)
            delta = (current_time - first_time).total_seconds() / 60.0
            t.append(delta)

    return {
        "t": t,
        "CGM": CGM,
        "BG": BG,
        "IOB": IOB,
        "CHO": CHO,
    }


def generate_plot_from_csv(csv_file, output_file=None, add_labels=False):
    """
    Generate plot from CSV data file.

    Args:
        csv_file: Path to CSV file
        output_file: Path for output SVG (default: same name in result/ folder)
        add_labels: Whether to add subplot labels (a, b)
    """
    csv_path = Path(csv_file)
    data = load_test_rig_data(csv_path)

    if output_file is None:
        output_file = result_dir / csv_path.with_suffix(".svg").name

    plot_bg_cho_iob_and_save(
        data["t"],
        data["CGM"],
        data["BG"],
        data["CHO"],
        data["IOB"],
        output_file,
        add_labels=add_labels,
    )
    print(f"Plot saved to: {output_file}")


def generate_merged_plot(csv_no_attack, csv_attack, output_file):
    """
    Generate a merged plot comparing attack vs no-attack CGM readings.

    Args:
        csv_no_attack: Path to no-attack CSV file (1st in pair)
        csv_attack: Path to attack CSV file (2nd in pair)
        output_file: Path for output SVG
    """
    data_no_attack = load_test_rig_data(csv_no_attack)
    data_attack = load_test_rig_data(csv_attack)

    plot_merged_and_save(data_no_attack, data_attack, output_file)
    print(f"Merged plot saved to: {output_file}")


def generate_all_plots():
    """Generate plots for all CSV files in the data/ folder."""
    # Track files used in merged plots
    grouped_files = set()

    # Generate merged plots for each pair (1st = no attack, 2nd = attack)
    for i, (no_attack_file, attack_file) in enumerate(csv_groups):
        no_attack_path = data_dir / no_attack_file
        attack_path = data_dir / attack_file
        grouped_files.add(no_attack_path)
        grouped_files.add(attack_path)

        output_file = result_dir / f"group_{i+1}.svg"
        print(f"Processing merged: {no_attack_file} + {attack_file}")
        generate_merged_plot(no_attack_path, attack_path, output_file)

    # Generate individual plots for non-grouped files
    csv_files = list(data_dir.glob("*.csv"))
    ungrouped_files = [f for f in csv_files if f not in grouped_files]

    for csv_file in sorted(ungrouped_files):
        print(f"Processing: {csv_file.name}")
        generate_plot_from_csv(csv_file, add_labels=True)


if __name__ == "__main__":
    generate_all_plots()
