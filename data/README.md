# Data Directory

This directory should contain the PhysioNet stress recognition dataset required for the simulation.

## Required File

Place your processed PhysioNet CSV file here as:
```
data/physionet_stress_data.csv
```

## Data Source

The required dataset can be obtained from:
- **Source:** Healey, J., & Picard, R. (2005). Detecting stress during real-world driving tasks using physiological sensors
- **URL:** https://physionet.org/content/drivedb/1.0.0/

## Required Columns

Your CSV file must contain the following columns:
- `HRV_LF_HF_ratio` - Heart rate variability low frequency to high frequency ratio
- `GSR_level` - Galvanic skin response measurements

## File Format

The CSV should contain preprocessed physiological data at 3-minute intervals for optimal simulation performance.
