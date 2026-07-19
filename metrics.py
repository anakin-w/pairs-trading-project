import numpy as np
from config import TRADING_DAYS


def annualized_sharpe(daily_returns, annualization=TRADING_DAYS):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan

    std_ret = daily_returns.std()
    if std_ret == 0:
        return np.nan

    return np.sqrt(annualization) * daily_returns.mean() / std_ret


def cagr(daily_returns, annualization=TRADING_DAYS):
    """Compound Annual Growth Rate: turn the total return into an average annual compound growth rate."""
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan

    total_years = len(daily_returns) / annualization
    final_equity = (1 + daily_returns).prod()
    return final_equity ** (1 / total_years) - 1


def max_drawdown(daily_returns):
    """How much did each point drop relative to its previous maximum value."""
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan

    equity_curve = (1 + daily_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return drawdown.min()


def total_return(daily_returns):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan

    return (1 + daily_returns).prod() - 1


def summarize_result(bt):
    """Compute all metrics and gather them into a dictionary."""
    net_ret = bt["net_ret"]
    num_trades = (bt["position_signals"].diff().abs().fillna(0) > 0).sum()

    return {
        "Sharpe": annualized_sharpe(net_ret, TRADING_DAYS),
        "CAGR": cagr(net_ret, TRADING_DAYS),
        "MaxDD": max_drawdown(net_ret),
        "TotalReturn": total_return(net_ret),
        "NumTrades": int(num_trades),
    }
