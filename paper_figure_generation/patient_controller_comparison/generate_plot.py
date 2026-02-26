"""
Generate plots from saved simulation data.

This script loads CSV data from the data/ folder and generates plots.
Adjust layout and styling here without re-running simulations.
"""

import csv
from pathlib import Path
from plot import TIRConfig, plot_bg_cho_iob_and_save, plot_comparison_and_save_with_tir

file_path = Path(__file__).resolve()
parent_folder = file_path.parent

data_dir = parent_folder / "data"
result_dir = parent_folder / "result"
result_dir.mkdir(parents=True, exist_ok=True)


def load_simulation_data(csv_file):
    """
    Load simulation data from CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        dict: Dictionary with keys 't', 'BG', 'IOB', 'CHO', 'INS', 'target'
    """
    t = []
    BG = []
    IOB = []
    CHO = []
    INS = []
    target = None

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if not row:  # Empty row separates data from metadata
                break
            t.append(float(row[0]))
            BG.append(float(row[1]))
            IOB.append(float(row[2]))
            CHO.append(float(row[3]))
            INS.append(float(row[4]))

        # Read metadata (target)
        for row in reader:
            if row and row[0] == "target":
                target = float(row[1])
                break

    return {
        "t": t,
        "BG": BG,
        "IOB": IOB,
        "CHO": CHO,
        "INS": INS,
        "target": target,
    }


def generate_plot_from_csv(csv_file, output_file=None):
    """
    Generate plot from CSV data file.

    Args:
        csv_file: Path to CSV file
        output_file: Path for output SVG (default: same name in result/ folder)
    """
    csv_path = Path(csv_file)
    data = load_simulation_data(csv_path)

    if output_file is None:
        output_file = result_dir / csv_path.with_suffix(".svg").name

    tir_config = TIRConfig()
    time_in_range = tir_config.calculate_time_in_range(data["BG"])

    plot_bg_cho_iob_and_save(
        data["t"],
        data["BG"],
        data["CHO"],
        data["IOB"],
        output_file,
        time_in_range,
        tir_config,
    )
    print(f"Plot saved to: {output_file}")


def generate_all_plots():
    """Generate plots for all CSV files in the data/ folder."""
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        generate_plot_from_csv(csv_file)


def generate_comparison_plot(csv_left, csv_right, output_file, title_left="", title_right=""):
    """
    Generate a side-by-side comparison plot from two CSV files.

    Args:
        csv_left: Path to left CSV file
        csv_right: Path to right CSV file
        output_file: Path for output SVG
        title_left: Title for left plot
        title_right: Title for right plot
    """
    data_left = load_simulation_data(csv_left)
    data_right = load_simulation_data(csv_right)

    tir_config = TIRConfig()
    data_left["time_in_range"] = tir_config.calculate_time_in_range(data_left["BG"])
    data_right["time_in_range"] = tir_config.calculate_time_in_range(data_right["BG"])

    plot_comparison_and_save_with_tir(
        data_left, data_right, output_file, tir_config, title_left, title_right
    )
    output_path = Path(output_file)
    print(f"Comparison plot saved to: {output_path.with_suffix('.svg')} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    # Generate comparison plot
    basal_csv = data_dir / "t1dm_adult#007_basal_bolus_75g_at_20min.csv"
    oref0_csv = data_dir / "t1dm_adult#007_oref0_bolus_75g_at_20min.csv"
    output = result_dir / "comparison_basal_vs_oref0.svg"

    generate_comparison_plot(
        basal_csv, oref0_csv, output,
        title_left="Basal-Bolus", title_right="OpenAPS oref0"
    )
