import yfinance as yf
from config import TICKER_Y, TICKER_X


def download_data(start_date, end_date):
    raw_data = yf.download([TICKER_Y, TICKER_X], start=start_date, end=end_date, auto_adjust=True, progress=False)

    close = raw_data["Close"].copy()
    close = close.dropna()

    return close
