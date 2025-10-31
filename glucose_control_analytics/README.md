# Glucose Control Analytics

A standalone Python package for glucose control analytics, providing tools for time-in-range calculations, visualization, and patient type definitions.

## Features

- **Time in Range (TIR) Calculations**: Support for multiple TIR standards (BASIC and CLINICAL)
- **Visualization**: Plot glucose, insulin, and carbohydrate data with optional TIR overlays
- **Patient Types**: Standardized patient type definitions for diabetes management
- **Performance Metrics**: Calculate RMSE, MAE, MSE, and other control metrics

## Package Structure

```
glucose_control_analytics/
├── __init__.py           # Package exports
├── patient_types.py      # PatientType enum definitions
├── time_in_range.py      # TIR calculation and configuration
├── plot.py              # Plotting and visualization functions
└── README.md            # This file
```

## Usage

### Basic Import

```python
from glucose_control_analytics import (
    TIRConfig,
    TIRCategory,
    TIRStandard,
    PatientType,
    plot_and_save_with_tir,
)
```

### Time in Range Calculation

```python
# Create TIR config with BASIC standard (4 categories)
tir_config = TIRConfig(standard=TIRStandard.BASIC)

# Calculate time in range from BG values
BG_values = [120, 150, 180, 200, 90, 70]
time_in_range = tir_config.calculate_time_in_range(BG_values)

# Result: {'very_high': 16.7, 'high': 33.3, 'target': 33.3, 'low': 16.7}
```

### Plotting with TIR

```python
from pathlib import Path

# Prepare data
t = [0, 5, 10, 15, 20, 25]  # Time in minutes
BG = [120, 150, 180, 200, 90, 70]  # Blood glucose (mg/dL)
CHO = [0, 50, 0, 0, 0, 0]  # Carbohydrates (g)
insulin = [0.5, 0.6, 0.7, 0.8, 0.5, 0.4]  # Insulin (U/min)
target_BG = 100  # Target blood glucose

# Calculate TIR
tir_config = TIRConfig()
time_in_range = tir_config.calculate_time_in_range(BG)

# Save plot with TIR visualization
plot_and_save_with_tir(
    t, BG, CHO, insulin, target_BG,
    file_name=Path("output.png"),
    time_in_range=time_in_range,
    tir_config=tir_config
)
```

### Patient Type Usage

```python
from glucose_control_analytics import PatientType, TIRConfig

# Check if TIR values are acceptable for a patient group
tir_config = TIRConfig(standard=TIRStandard.BASIC)
time_in_range = tir_config.calculate_time_in_range(BG_values)

results, count = tir_config.is_time_in_range_acceptable(
    time_in_range,
    PatientType.ADULT
)
```

## TIR Standards

### BASIC Standard (4 categories)
- **Very High**: > 250 mg/dL
- **High**: 180-250 mg/dL
- **Target**: 70-180 mg/dL
- **Low**: < 70 mg/dL

### CLINICAL Standard (5 categories)
- **Very High**: > 250 mg/dL
- **High**: 180-250 mg/dL
- **Target**: 70-180 mg/dL
- **Low**: 54-70 mg/dL
- **Very Low**: < 54 mg/dL

## Clinical Reference Ranges

Based on [NEJM 2022 Study](https://www.nejm.org/doi/full/10.1056/NEJMoa2203913):

| Patient Group | Very High (>250%) | High (180-250%) | Target (70-180%) | Low (<70%) |
|--------------|-------------------|-----------------|------------------|------------|
| Children (6-18) | 9.3±6.0 | 21.1±6.8 | 67.5±11.5 | 2.1±1.5 |
| Adults (18+) | 5.6±4.9 | 18.2±8.4 | 74.5±11.9 | 1.6±2.1 |

## Using in Other Repositories

To use this package in another repository:

1. **Add to Python path**:
```python
import sys
from pathlib import Path
sys.path.append(str(Path("/path/to/simglucose")))
```

2. **Import and use**:
```python
from glucose_control_analytics import TIRConfig, plot_and_save_with_tir
```

Alternatively, you can copy the entire `glucose_control_analytics` folder to your other repository.

## Dependencies

- `numpy`: For numerical calculations
- `matplotlib`: For plotting and visualization
- `enum`: For type definitions (Python standard library)

## Version

Current version: 1.0.0
