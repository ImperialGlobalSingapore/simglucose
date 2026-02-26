"""Temporary script to test standalone TIR pie chart."""

from pathlib import Path
import matplotlib.pyplot as plt
from plot import TIRConfig, TIRCategory, _plot_time_in_range_scale

result_dir = Path(__file__).resolve().parent / "result"
result_dir.mkdir(parents=True, exist_ok=True)

tir_config = TIRConfig()

# Data from basal-bolus simulation (left plot)
tir_left = {
    TIRCategory.TARGET: 88.8,
    TIRCategory.HIGH: 11.2,
}

# Data from oref0 simulation (right plot)
tir_right = {
    TIRCategory.TARGET: 87.2,
    TIRCategory.HIGH: 12.8,
}


def plot_tir_pie_and_save(time_in_range, tir_config, file_name):
    fig, ax = plt.subplots(figsize=(3, 3))
    _plot_time_in_range_scale(ax, time_in_range, tir_config)

    file_path = Path(file_name)
    fig.savefig(file_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(file_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"Saved: {file_path.with_suffix('.svg')} and {file_path.with_suffix('.png')}")


if __name__ == "__main__":
    plot_tir_pie_and_save(tir_left, tir_config, result_dir / "test_pie_basal")
