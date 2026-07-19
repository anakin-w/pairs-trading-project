import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from config import (TICKER_Y, TICKER_X, BETA_LOOKBACK_GRID, Z_LOOKBACK_GRID, ENTRY_Z_GRID, EXIT_Z_GRID, TRADING_DAYS)
from metrics import annualized_sharpe


def compute_beta(prices, beta_mode, beta_lookback=None):
    """
    Return a hedge-ratio (beta) series for Y = alpha + beta * X + error.
    - "fixed": one OLS beta on the whole slice, broadcast to a constant series.
    - "rolling": rolling single-variable OLS beta = Cov(X, Y) / Var(X), computed in a vectorized way.
    """
    y = prices[TICKER_Y]
    x = prices[TICKER_X]

    if beta_mode == "fixed":
        cov = np.cov(x.values, y.values)[0, 1]
        beta_value = cov / np.var(x.values)
        return pd.Series(beta_value, index=prices.index)

    if beta_mode == "rolling":
        cov = x.rolling(beta_lookback).cov(y)
        var = x.rolling(beta_lookback).var()
        return cov / var


def prepare_data(prices, beta):
    """Build the spread and daily-return columns given a beta series."""
    df = prices.copy()

    df["beta"] = beta
    df["spread"] = df[TICKER_Y] - df["beta"] * df[TICKER_X]

    df["ret_Y"] = df[TICKER_Y].pct_change().fillna(0)
    df["ret_X"] = df[TICKER_X].pct_change().fillna(0)

    return df


def generate_positions(z_scores, entry_z, exit_z):
    """Generate spread positions (long/short the spread, not a single asset) from z-scores."""
    position_list = []
    current_position = 0

    for z in z_scores.to_numpy():
        if np.isnan(z):
            current_position = 0
        elif current_position == 0:
            if z <= -entry_z:
                current_position = 1
            elif z >= entry_z:
                current_position = -1
        elif current_position == 1:
            # Exit only; don't flip on the same day to avoid whipsaw next day.
            if z >= -exit_z:
                current_position = 0
        elif current_position == -1:
            if z <= exit_z:
                current_position = 0

        position_list.append(current_position)

    return pd.Series(position_list, index=z_scores.index)


def backtest_strategy(prices, beta_mode, beta_lookback, z_lookback, entry_z, exit_z, cost_bps):
    """Backtest for either beta mode. beta_lookback is ignored when beta_mode == 'fixed'."""
    beta = compute_beta(prices, beta_mode, beta_lookback)
    df = prepare_data(prices, beta)

    rolling_mean = df["spread"].rolling(z_lookback).mean()
    rolling_std = df["spread"].rolling(z_lookback).std()
    df["z_scores"] = (df["spread"] - rolling_mean) / rolling_std

    df["position_signals"] = generate_positions(df["z_scores"], entry_z, exit_z)

    # Convert a "long/short spread" signal into Y/X (e.g. GLD/GDX) weights (works for both scalar and series beta).
    gross_exposure = 1 + df["beta"].abs()
    weight_y = 1 / gross_exposure
    weight_x = -df["beta"] / gross_exposure

    df["Y_position"] = df["position_signals"] * weight_y
    df["X_position"] = df["position_signals"] * weight_x

    df["Y_position_shifted"] = df["Y_position"].shift(1).fillna(0)
    df["X_position_shifted"] = df["X_position"].shift(1).fillna(0)

    df["pnl_rate"] = (df["Y_position_shifted"] * df["ret_Y"] + df["X_position_shifted"] * df["ret_X"])

    # Transaction (turnover) costs.
    df["turnover"] = ((df["Y_position"] - df["Y_position_shifted"]).abs() 
                      + (df["X_position"] - df["X_position_shifted"]).abs())
    df["cost"] = df["turnover"] * cost_bps / 10000.0

    df["net_ret"] = df["pnl_rate"] - df["cost"]
    df["equity_curve"] = (1 + df["net_ret"]).cumprod()

    return df


def _param_grid(beta_mode):
    """
    Yield (beta_lookback, z_lookback, entry_z, exit_z) tuples, skipping exit_z >= entry_z.
    In fixed mode beta_lookback is a placeholder (None) so downstream code is uniform.
    """
    beta_grid = BETA_LOOKBACK_GRID if beta_mode == "rolling" else [None]
    for beta_lb, z_lb, entry_z, exit_z in product(
            beta_grid, Z_LOOKBACK_GRID, ENTRY_Z_GRID, EXIT_Z_GRID):
        if exit_z >= entry_z:
            continue
        yield beta_lb, z_lb, entry_z, exit_z


def choose_best_params(train_prices, beta_mode, cost_bps, n_splits):
    # Ordinary CV would leak the future on time series, so use expanding-window splits.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(train_prices))

    # Check that the shortest fold's training segment can warm up the largest lookback window.
    max_lookback = max(Z_LOOKBACK_GRID)
    if beta_mode == "rolling":
        max_lookback = max(max_lookback, max(BETA_LOOKBACK_GRID))

    shortest_train = min(len(train_index) for train_index, _ in splits)
    if shortest_train < max_lookback:
        raise ValueError(f"Shortest CV training fold has {shortest_train} rows, but the largest lookback window needs {max_lookback}."
                         f"Increase TRAIN_RATIO, reduce N_SPLITS, or shrink the lookback grids.")

    all_results = []
    for beta_lb, z_lb, entry_z, exit_z in _param_grid(beta_mode):
        prefix_len = (max(beta_lb, z_lb) if beta_mode == "rolling" else z_lb) - 1

        sharpe_list = []
        for train_index, val_index in splits:
            fold_train = train_prices.iloc[train_index]
            fold_val = train_prices.iloc[val_index]

            # Rolling stats need warm-up, so prepend a prefix taken from fold_train.
            fold_val_with_prefix = pd.concat([fold_train.iloc[-prefix_len:], fold_val])

            bt_result = backtest_strategy(fold_val_with_prefix, beta_mode, beta_lb, z_lb, entry_z, exit_z, cost_bps)
            bt_result = bt_result.loc[fold_val.index]  # drop the prefix rows

            fold_sharpe = annualized_sharpe(bt_result["net_ret"], TRADING_DAYS)
            if not np.isnan(fold_sharpe):
                sharpe_list.append(fold_sharpe)

        row = {
            "beta_lookback": beta_lb,
            "z_lookback": z_lb,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "cv_mean_sharpe": np.mean(sharpe_list) if sharpe_list else np.nan,
        }
        all_results.append(row)

    cv_results = (pd.DataFrame(all_results)
                  .sort_values("cv_mean_sharpe", ascending=False)
                  .reset_index(drop=True))

    best_row = cv_results.iloc[0]
    best_params = {
        "beta_lookback": None if beta_mode == "fixed" else int(best_row["beta_lookback"]),
        "z_lookback": int(best_row["z_lookback"]),
        "entry_z": float(best_row["entry_z"]),
        "exit_z": float(best_row["exit_z"]),
    }

    # fixed mode: drop the all-None beta column for a cleaner table.
    if beta_mode == "fixed":
        cv_results = cv_results.drop(columns=["beta_lookback"])

    return best_params, cv_results
