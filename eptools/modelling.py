from typing import Any, Dict, Iterator
from eptools.data_preprocessing import get_all_months, get_nixtla_df


# forecast protocol constants — fixed by the project spec.
LAG_MONTHS = 3    # months between the last known observation and the scored forecast month
HORIZON    = 18   # full forecast curve length in months


def expanding_window_backtest_folds(
    min_train_months: int = 12,
) -> Iterator[Dict[str, Any]]:
    """
    Rolling-origin, expanding-window backtest generator matching the
    spec's fixed 3-month forecast lag and 18-month horizon.

    Training data always starts from the first available month and grows
    as the origin advances (expanding window) — nothing is dropped from
    the trailing end.

    At each origin month `t`, yields a dict with:
      - origin        : the origin month t (last month of "known" history)
      - target_date   : t + LAG_MONTHS — the single month this fold is scored on
      - horizon_dates : t+1 .. t+HORIZON — full forecast curve to produce,
                        even though only target_date gets scored
      - train_df      : nixtla-format df truncated to ds <= origin
                        (only data actually available at that origin)
      - test_df       : actuals for target_date only, held out separately
                        for scoring AFTER the model forecasts — never pass
                        this into training

    Parameters
    ----------
    min_train_months : int
        Minimum history required before the first origin is emitted.
        Not part of the spec — a tunable implementation choice for how
        much warm-up history the model needs.
    """
    # get all the months in our training data
    all_months = get_all_months()
    # get the nixtla formatted dataframe
    nixtla_df = get_nixtla_df()  # expects columns: unique_id, ds, y

    # keep track of where we will be making the forecast from in each fold
    # initialise this as the offset of minimum training months - 1 as this is the smallest first fold 
    first_origin_idx = min_train_months - 1
    
    # the last point we can make a forecast is at the end of the data minus the lag months
    # any further and we have nothing to test the forecast against
    last_origin_idx = len(all_months) - 1 - LAG_MONTHS

    # it's possible to set minimum training months so high that we don't have enough data to generate a fold
    # check for that
    if last_origin_idx < first_origin_idx:
        raise ValueError(
            f"Not enough months ({len(all_months)}) for "
            f"min_train_months={min_train_months} and LAG_MONTHS={LAG_MONTHS}."
        )

    # loop through all forecasting points / end-of-training-set points until we get to the highest value that permits a testable forecast
    for origin_idx in range(first_origin_idx, last_origin_idx + 1):
        
        # get the month the forecast will be made
        origin = all_months[origin_idx]
        # get the month that we need to minimise our metrics for
        target_date = all_months[origin_idx + LAG_MONTHS]

        # we are using an expanding window so grab everything in the dataset before the forecast point as training data
        train_df = nixtla_df.loc[nixtla_df["ds"] <= origin].copy()

        # we are required to make predictions out to an 18 month horizon
        # no value in predicting beyond the end of available data so don't go beyond that
        horizon_end_idx = min(origin_idx + HORIZON, len(all_months) - 1)
        horizon_dates = all_months[origin_idx + 1 : horizon_end_idx + 1]


        test_df = nixtla_df.loc[nixtla_df["ds"] == target_date].copy()

        yield {
            "origin": origin,
            "target_date": target_date,
            "horizon_dates": horizon_dates,
            "train_df": train_df,
            "test_df": test_df,
        }
