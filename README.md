# Pairs Trading Backtest

A mean-reversion pairs-trading backtest for a configurable pair of tickers (default: GLD / GDX).   
It estimates a hedge ratio (beta), trades the spread on z-score signals, selects hyperparameters via time-series cross-validation on a training set, and evaluates on a held-out test set.

## Strategy

For a pair `Y` and `X`, the strategy models `Y = alpha + beta * X + error` and trades the spread `Y - beta * X`:

- **Beta (hedge ratio)** is estimated either once on the full training slice (`fixed`) or on a rolling window (`rolling`).
- **Signals** come from the z-score of the spread over a rolling window. Enter when the z-score exceeds `entry_z`, exit when it reverts inside `exit_z`.
- **Positions** are converted into `Y`/`X` weights normalized by gross exposure, with turnover-based transaction costs applied.

## Project Structure

- `config.py` — All parameters: tickers, date range, train ratio, beta mode, hyperparameter grids, costs.
- `data.py` — Downloads adjusted close prices via `yfinance`.
- `strategy.py` — Beta estimation, z-score signals, backtest, and cross-validated parameter selection.
- `metrics.py` — Performance metrics: Sharpe, CAGR, max drawdown, total return, trade count.
- `main.py` — Entry point: download data → train/test split → CV → backtest on test set → cost robustness check.

## Configuration

Edit `config.py` to set the pair (`TICKER_Y`, `TICKER_X`) and parameters (`BETA_MODE`, `TRAIN_RATIO`, `COST_BPS`).  
Hyperparameters (`BETA_LOOKBACK_GRID`, `Z_LOOKBACK_GRID`, `ENTRY_Z_GRID`, `EXIT_Z_GRID`) are searched over via cross-validation.

## Usage

```bash
python main.py                    
python main.py --beta-mode fixed  
python main.py --beta-mode rolling
```

## Notes

- Cross-validation uses expanding-window `TimeSeriesSplit` to avoid look-ahead bias, and rolling statistics are warmed up with a training-set prefix on each fold and on the test set.
- Choose pairs with an economic or statistical basis (e.g. cointegration); arbitrary pairs may violate the mean-reversion assumption and produce meaningless results.