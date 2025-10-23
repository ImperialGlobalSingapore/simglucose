"""
File I/O utilities for glucose control data.

This module provides functions to save glucose control data to various file formats.
"""


def save_to_csv(log_dir, t, BG, CHO, insulin, file_name):
    """
    Save glucose control data to CSV file.

    Args:
        log_dir: Directory to save the CSV file (Path object or string)
        t: Time array
        BG: Blood glucose array
        CHO: Carbohydrate array
        insulin: Insulin array
        file_name: Base name for the CSV file (without extension)
    """
    csv_file = log_dir / f"{file_name}.csv"
    with open(csv_file, "w") as f:
        f.write("time,CHO,insulin,BG\n")
        for i in range(len(t)):
            f.write(f"{t[i]},{CHO[i]},{insulin[i]},{BG[i]}\n")
