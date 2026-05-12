#!/usr/bin/env python3
"""
Report-aligned reproduction script for:
Graph-Regularized Federated Learning for FMI Weather Stations

This script follows the final report settings:
- dataset file: fmi_daily_50stations_raw.csv
- chronological split: 2023 train / 2024 test
- features: tday, tmin, tmax, snow, sin_doy, cos_doy
- target: next-day mean temperature
- graph systems:
  1) geography-only kNN graph
  2) geography + positive training-period correlation graph
- k = 3
- sigma_km = 300
- alpha sweep = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
- learning rate = 1e-3
- 3000 iterations for sweep, 5000 for final fit
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# -----------------------------
# Configuration
# -----------------------------

DATA_FILENAME = "fmi_daily_50stations_raw.csv"
SPLIT_DATE = pd.Timestamp("2024-01-01")
FEATURE_COLS = ["tday", "tmin", "tmax", "snow", "sin_doy", "cos_doy"]
TARGET_COL = "target_tday_next"
K_NEIGHBORS = 3
SIGMA_KM = 300.0
LAMBDA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
LEARNING_RATE = 1e-3
SWEEP_ITERS = 3000
FINAL_ITERS = 5000
LAM_RIDGE = 1e-2

REPORT_REFERENCE = {
    "local": 3.109,
    "global": 2.905,
    "graph_geo": 3.119,
    "graph_geo_corr": 3.119,
    "best_lambda_geo": 1.0,
    "best_lambda_geo_corr": 1.0,
}


# -----------------------------
# Data loading and preprocessing
# -----------------------------

def load_and_prepare_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"Place '{DATA_FILENAME}' in the same folder as this script or update DATA_FILENAME."
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    # Seasonal features
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # Next-day target
    df[TARGET_COL] = df.groupby("station_id")["tday"].shift(-1)

    # Drop rows that cannot be used
    model_df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    model_df = model_df.sort_values(["station_id", "date"]).reset_index(drop=True)
    return model_df


def split_train_test(model_df: pd.DataFrame):
    train_df = model_df[model_df["date"] < SPLIT_DATE].copy()
    test_df = model_df[model_df["date"] >= SPLIT_DATE].copy()

    common_station_ids = sorted(
        set(train_df["station_id"].unique()) & set(test_df["station_id"].unique())
    )

    train_df = train_df[train_df["station_id"].isin(common_station_ids)].copy()
    test_df = test_df[test_df["station_id"].isin(common_station_ids)].copy()
    model_df = model_df[model_df["station_id"].isin(common_station_ids)].copy()

    stations_df = (
        model_df.sort_values(["station_id", "date"])
        .groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .sort_values("station_id")
        .reset_index(drop=True)
    )

    return train_df, test_df, stations_df


def build_station_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, station_ids: List[int]):
    train_data = {}
    test_data = {}

    for sid in station_ids:
        g_train = train_df[train_df["station_id"] == sid].sort_values("date")
        g_test = test_df[test_df["station_id"] == sid].sort_values("date")

        X_train = g_train[FEATURE_COLS].to_numpy(dtype=float)
        y_train = g_train[TARGET_COL].to_numpy(dtype=float)

        X_test = g_test[FEATURE_COLS].to_numpy(dtype=float)
        y_test = g_test[TARGET_COL].to_numpy(dtype=float)

        train_data[sid] = (X_train, y_train)
        test_data[sid] = (X_test, y_test)

    return train_data, test_data


def standardize_and_add_intercept(train_data, test_data, station_ids):
    X_train_all = np.vstack([train_data[sid][0] for sid in station_ids])

    mean_x = X_train_all.mean(axis=0)
    std_x = X_train_all.std(axis=0)
    std_x[std_x == 0] = 1.0

    for sid in station_ids:
        X_train, y_train = train_data[sid]
        X_test, y_test = test_data[sid]

        X_train = (X_train - mean_x) / std_x
        X_test = (X_test - mean_x) / std_x

        # Add intercept after standardization
        X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

        train_data[sid] = (X_train, y_train)
        test_data[sid] = (X_test, y_test)

    return train_data, test_data, mean_x, std_x


# -----------------------------
# Graph helpers
# -----------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def build_distance_matrix(stations_df: pd.DataFrame) -> np.ndarray:
    n = len(stations_df)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist_mat[i, j] = haversine_km(
                stations_df.loc[i, "lat"], stations_df.loc[i, "lon"],
                stations_df.loc[j, "lat"], stations_df.loc[j, "lon"],
            )
    return dist_mat


def build_graph_knn(
    stations_df: pd.DataFrame,
    dist_mat: np.ndarray,
    k: int = K_NEIGHBORS,
    sigma_km: float = SIGMA_KM,
    corr_mat: pd.DataFrame | None = None,
):
    edge_set = set()
    n = len(stations_df)

    for i in range(n):
        nn_idx = np.argsort(dist_mat[i])[:k + 1]
        nn_idx = [j for j in nn_idx if j != i][:k]
        for j in nn_idx:
            a, b = sorted((i, j))
            edge_set.add((a, b))

    edges = []
    for i, j in sorted(edge_set):
        dist = dist_mat[i, j]
        base_weight = float(np.exp(-(dist ** 2) / (sigma_km ** 2)))

        if corr_mat is None:
            weight = base_weight
        else:
            sid_i = stations_df.loc[i, "station_id"]
            sid_j = stations_df.loc[j, "station_id"]
            corr_ij = 0.0
            if sid_i in corr_mat.index and sid_j in corr_mat.columns:
                corr_ij = max(float(corr_mat.loc[sid_i, sid_j]), 0.0)
            weight = base_weight * corr_ij

        edges.append(
            {
                "i_idx": i,
                "j_idx": j,
                "station_id_i": stations_df.loc[i, "station_id"],
                "station_id_j": stations_df.loc[j, "station_id"],
                "station_name_i": stations_df.loc[i, "station_name"],
                "station_name_j": stations_df.loc[j, "station_name"],
                "distance_km": dist,
                "weight": weight,
            }
        )

    edges_df = pd.DataFrame(edges)
    edge_list = [
        (int(row["i_idx"]), int(row["j_idx"]), float(row["weight"]))
        for _, row in edges_df.iterrows()
    ]
    return edges_df, edge_list


# -----------------------------
# Models and evaluation
# -----------------------------

def fit_graph_ridge_stable(
    train_data,
    station_ids,
    station_to_idx,
    edge_list,
    lam_graph=1.0,
    lam_ridge=LAM_RIDGE,
    lr=LEARNING_RATE,
    n_iter=SWEEP_ITERS,
    verbose=False,
):
    n_clients = len(station_ids)
    d = next(iter(train_data.values()))[0].shape[1]
    W = np.zeros((n_clients, d), dtype=float)

    ridge_mask = np.ones(d, dtype=float)
    ridge_mask[-1] = 0.0  # do not ridge-regularize intercept

    for it in range(n_iter):
        grad = np.zeros_like(W)

        for sid in station_ids:
            idx = station_to_idx[sid]
            X, y = train_data[sid]
            n_i = len(y)

            pred = X @ W[idx]
            resid = pred - y
            grad[idx] += (2.0 / n_i) * (X.T @ resid) + 2.0 * lam_ridge * (ridge_mask * W[idx])

        for i, j, aij in edge_list:
            diff = W[i] - W[j]
            grad[i] += 2.0 * lam_graph * aij * diff
            grad[j] -= 2.0 * lam_graph * aij * diff

        grad_norm = np.linalg.norm(grad)
        if not np.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at iteration {it}")

        W -= lr * grad

    return W


def evaluate_weight_matrix(W, test_data, station_ids, station_to_idx, stations_df, colname):
    rows = []
    for sid in station_ids:
        idx = station_to_idx[sid]
        X_test, y_test = test_data[sid]
        pred = X_test @ W[idx]
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        station_name = stations_df.loc[stations_df["station_id"] == sid, "station_name"].iloc[0]
        rows.append({"station_id": sid, "station_name": station_name, colname: rmse})
    return pd.DataFrame(rows)


def run_local_baseline(train_data, test_data, station_ids, stations_df):
    rows = []
    for sid in station_ids:
        X_train, y_train = train_data[sid]
        X_test, y_test = test_data[sid]

        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        station_name = stations_df.loc[stations_df["station_id"] == sid, "station_name"].iloc[0]
        rows.append({"station_id": sid, "station_name": station_name, "rmse_local": rmse})

    return pd.DataFrame(rows)


def run_global_baseline(train_data, test_data, station_ids, stations_df):
    X_train_global = np.vstack([train_data[sid][0] for sid in station_ids])
    y_train_global = np.concatenate([train_data[sid][1] for sid in station_ids])

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train_global, y_train_global)

    rows = []
    for sid in station_ids:
        X_test, y_test = test_data[sid]
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        station_name = stations_df.loc[stations_df["station_id"] == sid, "station_name"].iloc[0]
        rows.append({"station_id": sid, "station_name": station_name, "rmse_global": rmse})

    return pd.DataFrame(rows), model


def run_sweep(edge_list, train_data, test_data, station_ids, station_to_idx):
    rows = []
    for lam in LAMBDA_VALUES:
        W = fit_graph_ridge_stable(
            train_data=train_data,
            station_ids=station_ids,
            station_to_idx=station_to_idx,
            edge_list=edge_list,
            lam_graph=lam,
            lam_ridge=LAM_RIDGE,
            lr=LEARNING_RATE,
            n_iter=SWEEP_ITERS,
            verbose=False,
        )

        rmses = []
        for sid in station_ids:
            idx = station_to_idx[sid]
            X_test, y_test = test_data[sid]
            pred = X_test @ W[idx]
            rmses.append(np.sqrt(mean_squared_error(y_test, pred)))

        rows.append({"lambda": lam, "mean_rmse": float(np.mean(rmses))})
    return pd.DataFrame(rows)


def compare_to_report(summary_df, best_geo_lambda, best_geocorr_lambda):
    print("\nValidation against report reference values")
    print("-" * 48)
    for _, row in summary_df.iterrows():
        model = row["model"]
        actual = float(row["mean_rmse"])
        ref = REPORT_REFERENCE.get(model)
        if ref is None:
            continue
        print(f"{model:>16}: actual={actual:.3f} | report={ref:.3f} | diff={actual-ref:+.3f}")

    print(f"{'best_lambda_geo':>16}: actual={best_geo_lambda:.2f} | report={REPORT_REFERENCE['best_lambda_geo']:.2f}")
    print(f"{'best_lambda_geo_corr':>16}: actual={best_geocorr_lambda:.2f} | report={REPORT_REFERENCE['best_lambda_geo_corr']:.2f}")


def main():
    model_df = load_and_prepare_data(DATA_FILENAME)
    train_df, test_df, stations_df = split_train_test(model_df)

    station_ids = stations_df["station_id"].tolist()
    station_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}

    train_data, test_data = build_station_datasets(train_df, test_df, station_ids)
    train_data, test_data, mean_x, std_x = standardize_and_add_intercept(train_data, test_data, station_ids)

    local_results = run_local_baseline(train_data, test_data, station_ids, stations_df)
    global_results, global_model = run_global_baseline(train_data, test_data, station_ids, stations_df)

    dist_mat = build_distance_matrix(stations_df)
    edges_geo_df, edge_list_geo = build_graph_knn(stations_df, dist_mat, k=K_NEIGHBORS, sigma_km=SIGMA_KM, corr_mat=None)

    pivot_train = train_df.pivot_table(index="date", columns="station_id", values="tday").sort_index()
    corr_mat = pivot_train.corr().fillna(0.0)
    edges_geocorr_df, edge_list_geocorr = build_graph_knn(
        stations_df, dist_mat, k=K_NEIGHBORS, sigma_km=SIGMA_KM, corr_mat=corr_mat
    )

    sweep_geo = run_sweep(edge_list_geo, train_data, test_data, station_ids, station_to_idx)
    sweep_geocorr = run_sweep(edge_list_geocorr, train_data, test_data, station_ids, station_to_idx)

    best_geo_lambda = float(sweep_geo.loc[sweep_geo["mean_rmse"].idxmin(), "lambda"])
    best_geocorr_lambda = float(sweep_geocorr.loc[sweep_geocorr["mean_rmse"].idxmin(), "lambda"])

    W_geo = fit_graph_ridge_stable(
        train_data=train_data,
        station_ids=station_ids,
        station_to_idx=station_to_idx,
        edge_list=edge_list_geo,
        lam_graph=best_geo_lambda,
        lam_ridge=LAM_RIDGE,
        lr=LEARNING_RATE,
        n_iter=FINAL_ITERS,
        verbose=False,
    )
    W_geocorr = fit_graph_ridge_stable(
        train_data=train_data,
        station_ids=station_ids,
        station_to_idx=station_to_idx,
        edge_list=edge_list_geocorr,
        lam_graph=best_geocorr_lambda,
        lam_ridge=LAM_RIDGE,
        lr=LEARNING_RATE,
        n_iter=FINAL_ITERS,
        verbose=False,
    )

    graph_results_geo = evaluate_weight_matrix(W_geo, test_data, station_ids, station_to_idx, stations_df, "rmse_graph_geo")
    graph_results_geocorr = evaluate_weight_matrix(W_geocorr, test_data, station_ids, station_to_idx, stations_df, "rmse_graph_geo_corr")

    comparison = (
        local_results
        .merge(global_results, on=["station_id", "station_name"], how="inner")
        .merge(graph_results_geo, on=["station_id", "station_name"], how="inner")
        .merge(graph_results_geocorr, on=["station_id", "station_name"], how="inner")
    )

    summary = pd.DataFrame({
        "model": ["local", "global", "graph_geo", "graph_geo_corr"],
        "mean_rmse": [
            comparison["rmse_local"].mean(),
            comparison["rmse_global"].mean(),
            comparison["rmse_graph_geo"].mean(),
            comparison["rmse_graph_geo_corr"].mean(),
        ],
    })

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    summary.to_csv(output_dir / "summary_report_aligned.csv", index=False)
    comparison.to_csv(output_dir / "comparison_report_aligned.csv", index=False)
    sweep_geo.to_csv(output_dir / "sweep_geo.csv", index=False)
    sweep_geocorr.to_csv(output_dir / "sweep_geo_corr.csv", index=False)
    edges_geo_df.to_csv(output_dir / "edges_geo.csv", index=False)
    edges_geocorr_df.to_csv(output_dir / "edges_geo_corr.csv", index=False)

    print("\nTrain date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Test date range: ", test_df["date"].min(), "to", test_df["date"].max())
    print("Number of stations:", len(station_ids))
    print("\nSummary:")
    print(summary)
    compare_to_report(summary, best_geo_lambda, best_geocorr_lambda)
    print("\nSaved outputs to:", output_dir.resolve())


if __name__ == "__main__":
    main()
