# Graph-Regularized Federated Learning for FMI Weather Stations

This project studies graph-based federated learning on Finnish Meteorological Institute weather-station data. Each station is treated as one client node, and the task is next-day temperature prediction from daily weather observations.

## Overview
The project compares:
- local linear models trained independently at each station
- one pooled global linear model
- a graph-regularized federated learning model using a geography-only graph
- a graph-regularized federated learning model using geography plus temperature-correlation weighting

## Data
- Source: FMI Open Data
- Stations: 42 weather stations
- Time range: 2023-01-01 to 2024-12-31
- Train/test split: 2023 for training, 2024 for testing

## Features and Target
Features:
- `tday`
- `tmin`
- `tmax`
- `snow`
- `sin(doy)`
- `cos(doy)`

Target:
- next-day mean temperature

## Main Result
In this setup, the pooled global linear model achieved the best mean RMSE. The graph-regularized personalized models were stable but did not outperform the global baseline.

## Methods
- local linear regression
- pooled global linear regression
- smooth graph-regularized optimization
- geography-only and geography+correlation graph construction

## Tools
- Python
- pandas
- numpy
- scikit-learn
- matplotlib

## Repository structure
- `notebooks/` or main notebook/script for experiments
- `data/` for processed files if included
- `report/` for course report and slides if included

## Notes
This project was completed as part of a federated learning course project.
