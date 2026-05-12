# Graph-Regularized Federated Learning for FMI Weather Stations

This repository contains a report-aligned reproduction file for the project **Graph-Regularized Federated Learning for FMI Weather Stations**.

## Included files

- `fl_weather_graph_learning.py` — clean Python script aligned with the final report
- `requirements.txt` — minimal dependencies
- `README.md` — setup and usage instructions

## Expected dataset

Place the dataset file below in the same folder as the script or notebook:

- `fmi_daily_50stations_raw.csv`

The script assumes this CSV contains at least:

- `date`
- `station_id`
- `station_name`
- `lat`
- `lon`
- `tday`
- `tmin`
- `tmax`
- `snow`

## Report-aligned settings

- Train/test split: **2023 train / 2024 test**
- Features:
  - `tday`
  - `tmin`
  - `tmax`
  - `snow`
  - `sin_doy`
  - `cos_doy`
- Target:
  - next-day mean temperature
- Graph systems:
  - geography-only kNN graph
  - geography + positive training-period correlation graph
- Graph hyperparameters:
  - `k = 3`
  - `sigma_km = 300`
- Optimization:
  - `learning_rate = 1e-3`
  - `3000` iterations for sweeps
  - `5000` iterations for final fits
- Alpha sweep:
  - `[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]`

## How to run

### Script
```bash
python fl_project_report_aligned.py
```

### Notebook
Open:
- `FL_Project_Report_Aligned.ipynb`

and run cells top to bottom.

## What gets saved

The script creates a `results/` folder containing:

- `summary_report_aligned.csv`
- `comparison_report_aligned.csv`
- `sweep_geo.csv`
- `sweep_geo_corr.csv`
- `edges_geo.csv`
- `edges_geo_corr.csv`

## Validation note

This script also prints a simple comparison between the computed summary values and the reference values reported in the final report:

- local: `3.109`
- global: `2.905`
- graph_geo: `3.119`
- graph_geo_corr: `3.119`

If your dataset matches the one used for the final report, the values should be close.
If they differ, the script will print the differences explicitly.
