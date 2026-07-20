import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from config import TICKER_Y, TICKER_X


def engle_granger_test(prices):
    """
    Engle-Granger cointegration test on a price slice.
    - Null hypothesis (H_0): the two series are NOT cointegrated. 
      Thus a small p-value is evidence FOR cointegration.
    - Returns a dict with the test statistic, p-value, and the 1/5/10% critical values.
    """
    y = prices[TICKER_Y]
    x = prices[TICKER_X]

    eg_stat, eg_pvalue, eg_crit = coint(y, x)

    return {
        "eg_stat": eg_stat,
        "eg_pvalue": eg_pvalue,
        "eg_crit_1pct": eg_crit[0],
        "eg_crit_5pct": eg_crit[1],
        "eg_crit_10pct": eg_crit[2],
    }


def rolling_cointegration(prices, window, step):
    """
    Slide a window across the sample and record the Engle-Granger p-value in each window. 
    Returns a DataFrame indexed by each window's end date, with the p-value.
    "step" lets us skip days to keep it fast (step=5 -> weekly).
    """
    y = prices[TICKER_Y]
    x = prices[TICKER_X]

    records = []
    for end in range(window, len(prices) + 1, step):
        start = end - window
        y_win = y.iloc[start:end]
        x_win = x.iloc[start:end]
        try:
            _, pvalue, _ = coint(y_win, x_win)
        except Exception:
            pvalue = np.nan
        records.append({"date": prices.index[end - 1], "eg_pvalue": pvalue})

    return pd.DataFrame(records).set_index("date")


def interpret_pvalue(pvalue):
    """Turn a p-value into a human-readable conclusion."""
    if np.isnan(pvalue):
        return "undetermined (test failed)"
    if pvalue < 0.01:
        return "strong evidence of cointegration (p < 1%)"
    if pvalue < 0.05:
        return "cointegrated at the 5% level"
    if pvalue < 0.10:
        return "weak/borderline (only significant at 10%)"
    return "no evidence of cointegration"


def summarize_cointegration(prices, rolling_window=None, rolling_step=5):
    """
    Convenience wrapper for main.py: run the full-slice Engle-Granger test and,
    if a rolling_window is given, also compute how often the pair is
    cointegrated across rolling windows.
    """
    # Test whether cointegration exists on the entire training set.
    result = engle_granger_test(prices)
    result["conclusion"] = interpret_pvalue(result["eg_pvalue"])

    # Test whether cointegration exists within each window. 
    if rolling_window is not None and len(prices) >= rolling_window:
        roll = rolling_cointegration(prices, rolling_window, step=rolling_step)
        frac = (roll["eg_pvalue"] < 0.05).mean()    # the percentage of windows that achieves 5% significant coint
        result["rolling_window"] = rolling_window
        result["rolling_frac_coint_5pct"] = frac
        result["rolling_n_windows"] = len(roll)

    return result
