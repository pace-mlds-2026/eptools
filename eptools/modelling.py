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
      - actuals_df    : actuals for target_date only, held out separately
                        for scoring AFTER the model forecasts — never pass
                        this into training

    Parameters
    ----------
    min_train_months : int
        Minimum history required before the first origin is emitted.
        Not part of the spec — a tunable implementation choice for how
        much warm-up history the model needs.
    """
    all_months = get_all_months()
    nixtla_df = get_nixtla_df()  # expects columns: unique_id, ds, y

    first_origin_idx = min_train_months - 1
    last_origin_idx = len(all_months) - 1 - LAG_MONTHS

    if last_origin_idx < first_origin_idx:
        raise ValueError(
            f"Not enough months ({len(all_months)}) for "
            f"min_train_months={min_train_months} and LAG_MONTHS={LAG_MONTHS}."
        )

    for origin_idx in range(first_origin_idx, last_origin_idx + 1):
        origin = all_months[origin_idx]
        target_date = all_months[origin_idx + LAG_MONTHS]

        # Expanding window: no lower bound on ds, only the upper cutoff moves.
        train_df = nixtla_df.loc[nixtla_df["ds"] <= origin].copy()

        horizon_end_idx = min(origin_idx + HORIZON, len(all_months) - 1)
        horizon_dates = all_months[origin_idx + 1 : horizon_end_idx + 1]

        actuals_df = nixtla_df.loc[nixtla_df["ds"] == target_date].copy()

        yield {
            "origin": origin,
            "target_date": target_date,
            "horizon_dates": horizon_dates,
            "train_df": train_df,
            "actuals_df": actuals_df,
        }
