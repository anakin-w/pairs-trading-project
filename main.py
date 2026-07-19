import argparse
import pandas as pd
from config import (TICKER_Y, TICKER_X, START_DATE, END_DATE, TRAIN_RATIO, BETA_MODE, COST_BPS, N_SPLITS)
from data import download_data
from strategy import backtest_strategy, choose_best_params
from metrics import summarize_result


def main(beta_mode):
    # -------------- 1. Download Data -------------------
    close_prices = download_data(START_DATE, END_DATE)
    print(f"Ticker_Y: {TICKER_Y}")
    print(f"Ticker_X: {TICKER_X}")
    print("Data range:", close_prices.index.min().date(), "→", close_prices.index.max().date(), "\n")

    # ------------- 2. Train/Test Split -----------------
    split_index = int(len(close_prices) * TRAIN_RATIO)
    train_prices = close_prices.iloc[:split_index]
    test_prices = close_prices.iloc[split_index:]

    print(f"Beta mode: {beta_mode}")
    print("Train size:", len(train_prices))
    print("Test size:", len(test_prices), "\n")

    # ------ 3. Cross-validation on Training Set --------
    best_params, cv_results = choose_best_params(train_prices, beta_mode, COST_BPS, N_SPLITS)

    print("Top 5 hyperparameter combinations on training set:")
    print(cv_results.head(5), "\n")
    print("Best params:")
    print(best_params, "\n")

    beta_lookback = best_params["beta_lookback"]
    z_lookback = best_params["z_lookback"]
    entry_z = best_params["entry_z"]
    exit_z = best_params["exit_z"]

    # --------- 4. Final Backtest on Test Set -----------
    # Prepend a training-set prefix so rolling stats have warm-up on the test set.
    prefix_len = (max(beta_lookback, z_lookback) if beta_mode == "rolling" else z_lookback) - 1
    test_prices_with_prefix = pd.concat([train_prices.iloc[-prefix_len:], test_prices])

    # Package "backtesting" and "removing the prefix".
    def run(cost):
        bt = backtest_strategy(test_prices_with_prefix, beta_mode, beta_lookback, z_lookback, entry_z, exit_z, cost)
        return bt.loc[test_prices.index]

    summary = summarize_result(run(COST_BPS))

    print("═" * 50)
    print("TEST SET PERFORMANCE:")
    print("═" * 50)
    for key, value in summary.items():
        print(key, ":", value)
    print("═" * 50)

    # ---- 5. Robustness Test accross Different TCs -----
    print("Robustness check under different transaction costs:")
    for cost in [0.0, 5.0, 10.0]:
        s = summarize_result(run(cost))

        print(f"Cost = {cost:>4.1f} bps | "
              f"Sharpe = {s['Sharpe']:.4f} | "
              f"CAGR = {s['CAGR']:.4f} | "
              f"MaxDD = {s['MaxDD']:.4f} | "
              f"Trades = {s['NumTrades']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairs trading backtest.")
    parser.add_argument("--beta-mode", choices=["fixed", "rolling"], default=BETA_MODE,
                        help="Hedge-ratio mode. Defaults to BETA_MODE in config.py.")
    args = parser.parse_args()
    main(args.beta_mode)
