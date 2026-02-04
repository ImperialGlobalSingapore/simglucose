"""
Generate plots from test rig CSV data.

This script loads CSV data from the data/ folder and generates plots.
Adjust layout and styling here without re-running simulations.
"""

import csv
from datetime import datetime
from pathlib import Path
from plot import plot_bg_cho_iob_and_save

file_path = Path(__file__).resolve()
parent_folder = file_path.parent

data_dir = parent_folder / "data"
result_dir = parent_folder / "result"
result_dir.mkdir(parents=True, exist_ok=True)


def load_test_rig_data(csv_file):
    """
    Load test rig data from CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        dict: Dictionary with keys 't', 'BG', 'IOB', 'CHO'
    """
    timestamps = []
    BG = []
    IOB = []
    CHO = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamps.append(row["timestamp"])
            BG.append(float(row["CGM_reading"]))
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
        "BG": BG,
        "IOB": IOB,
        "CHO": CHO,
    }


def generate_plot_from_csv(csv_file, output_file=None):
    """
    Generate plot from CSV data file.

    Args:
        csv_file: Path to CSV file
        output_file: Path for output SVG (default: same name in result/ folder)
    """
    csv_path = Path(csv_file)
    data = load_test_rig_data(csv_path)

    if output_file is None:
        output_file = result_dir / csv_path.with_suffix(".svg").name

    plot_bg_cho_iob_and_save(
        data["t"],
        data["BG"],
        data["CHO"],
        data["IOB"],
        output_file,
    )
    print(f"Plot saved to: {output_file}")


def generate_all_plots():
    """Generate plots for all CSV files in the data/ folder."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    for csv_file in sorted(csv_files):
        print(f"Processing: {csv_file.name}")
        generate_plot_from_csv(csv_file)


if __name__ == "__main__":
    generate_all_plots()
