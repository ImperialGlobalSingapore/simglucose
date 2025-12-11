"""
Glucose Control Analytics Package

A standalone package for glucose control analytics, providing tools for:
- Time in Range (TIR) calculations with multiple standards
- Plotting and visualization of glucose, insulin, and carbohydrate data
- Patient type definitions for diabetes management
- Statistical metrics for control performance evaluation
- File I/O utilities for saving data
- Parallel simulation framework for closed-loop testing

This package can be used independently with any glucose monitoring and control system.
"""

from .patient_types import PatientType
from .time_in_range import TIRCategory, TIRStandard, TIRConfig
from .plot import (
    plot_and_show,
    plot_and_show_with_tir,
    plot_and_save,
    plot_and_save_with_tir,
    plot_bg_cho_iob_and_show,
    plot_bg_cho_iob_and_show_with_tir,
    plot_bg_cho_iob_and_save,
    plot_bg_cho_iob_and_save_with_tir,
)
from .metrics import (
    get_rmse,
    get_mae,
    get_mse,
    get_max_error,
    get_iae,
    get_ise,
)
from .file_io import save_to_csv
from .openaps_parameter_tuning import OpenAPSParameterTuningBase

__all__ = [
    # Patient types
    "PatientType",
    # Time in range
    "TIRCategory",
    "TIRStandard",
    "TIRConfig",
    # Plotting functions
    "plot_and_show",
    "plot_and_show_with_tir",
    "plot_and_save",
    "plot_and_save_with_tir",
    "plot_bg_cho_iob_and_show",
    "plot_bg_cho_iob_and_show_with_tir",
    "plot_bg_cho_iob_and_save",
    "plot_bg_cho_iob_and_save_with_tir",
    # Metrics
    "get_rmse",
    "get_mae",
    "get_mse",
    "get_max_error",
    "get_iae",
    "get_ise",
    # File I/O
    "save_to_csv",
    # Parallel simulation
    "OpenAPSParameterTuningBase",
]

__version__ = "1.0.0"
