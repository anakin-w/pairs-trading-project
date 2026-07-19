TICKER_Y = "GLD"
TICKER_X = "GDX"

START_DATE = "2010-01-01"
END_DATE = None
TRAIN_RATIO = 0.7

# "fixed": estimate one beta on the whole training set.
# "rolling": re-estimate beta on a moving window.
BETA_MODE = "rolling"

# Rolling window for beta (has more days than the z-score window).
# Only used when BETA_MODE == "rolling".
BETA_LOOKBACK_GRID = [120, 150, 180, 200]

Z_LOOKBACK_GRID = [20, 40, 60, 90]
ENTRY_Z_GRID = [1.0, 1.5, 2.0, 2.5]
EXIT_Z_GRID = [0.2, 0.5, 1.0]

COST_BPS = 5.0
N_SPLITS = 5

TRADING_DAYS = 252
