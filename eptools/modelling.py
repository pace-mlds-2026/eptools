from typing import Any, Dict, Iterator, Optional
import pandas as pd
from eptools.data_preprocessing import get_all_months, get_nixtla_df



# forecast protocol constants — fixed by the project spec.
LAG_MONTHS = 3    # months between the last known observation and the scored forecast month
HORIZON    = 18   # full forecast curve length in months


# ── Scoring-universe helpers ─────────────────────────────────────────────────
# These describe *who exists, and when* — the panel's launch/discontinuation
# boundaries. They must be computed from a within-window-dense frame (see
# data_preprocessing.rebuild_nixtla_df) so the boundaries are real rather than an
# artefact of a global zero-fill.

def get_sku_active_windows(nixtla_df: pd.DataFrame) -> pd.DataFrame:
    """One row per series: first_seen, last_seen, window width, observed count, gaps."""
    w = (nixtla_df.groupby("unique_id")["ds"]
         .agg(first_seen="min", last_seen="max", n_rows_observed="count")
         .reset_index())
    w["window_months"] = (
        (w["last_seen"].dt.year - w["first_seen"].dt.year) * 12
        + (w["last_seen"].dt.month - w["first_seen"].dt.month) + 1
    )
    w["n_gap_months"] = w["window_months"] - w["n_rows_observed"]
    return w


def active_skus_at(active_windows: pd.DataFrame, target_date) -> set:
    """Series whose active window contains target_date."""
    mask = ((active_windows["first_seen"] <= target_date)
            & (active_windows["last_seen"] >= target_date))
    return set(active_windows.loc[mask, "unique_id"])


def build_full_panel(nixtla_df: pd.DataFrame, active_windows: pd.DataFrame) -> pd.DataFrame:
    """Reindex every series onto its own [first_seen, last_seen] range, zero-filling gaps.
    Within-window only, never the global calendar. Loud assertion on the row count so a
    future data refresh that isn't dense fails here rather than leaking into scoring."""
    full_index = pd.concat(
        [pd.DataFrame({"unique_id": r.unique_id,
                       "ds": pd.date_range(r.first_seen, r.last_seen, freq="MS")})
         for r in active_windows.itertuples()],
        ignore_index=True,
    )
    full = full_index.merge(nixtla_df[["unique_id", "ds", "y"]],
                            on=["unique_id", "ds"], how="left")
    full["y"] = full["y"].fillna(0.0)
    expected = int(active_windows["window_months"].sum())
    assert len(full) == expected, (
        f"build_full_panel row mismatch: expected {expected:,}, got {len(full):,}"
    )
    return full.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def train_frame(nixtla_df: pd.DataFrame, origin, window_months: Optional[int] = None) -> pd.DataFrame:
    """The single leakage boundary. Rows with ds <= origin (expanding window), or the
    trailing window_months (sliding window). Every feature/fit for an origin must come
    from this function's output, never from the full frame directly. The assertion must
    never fire; if it does, a future-dated row leaked or origin was miscomputed upstream."""
    if window_months is None:
        window_start = nixtla_df["ds"].min()
    else:
        window_start = origin - pd.DateOffset(months=window_months - 1)
    tf = nixtla_df[(nixtla_df["ds"] >= window_start) & (nixtla_df["ds"] <= origin)].copy()
    assert tf["ds"].max() <= origin, (
        f"LEAKAGE: train_frame returned data beyond origin {origin}"
    )
    return tf.reset_index(drop=True)



def expanding_window_backtest_folds(
    min_train_months: int = 12,
    window_months: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Rolling-origin backtest generator matching the spec's fixed 3-month forecast lag
    and 18-month horizon.

    By default the training window is *expanding* — it starts at the first available
    month and grows as the origin advances. Passing an integer `window_months` switches
    to a *sliding* window of that many trailing months (the structural-decline sweep the
    modelling notes flag). Either way the truncation goes through `train_frame`, the one
    leakage boundary.

    At each origin month `t`, yields a dict with:
      - origin        : the origin month t (last month of "known" history)
      - target_date   : t + LAG_MONTHS — the single month this fold is scored on
      - horizon_dates : t+1 .. t+HORIZON — full forecast curve to produce,
                        even though only target_date gets scored
      - train_df      : nixtla-format df truncated at the origin (see window_months)
      - test_df       : actuals for target_date only, held out separately for scoring
                        AFTER the model forecasts — never pass this into training

    Parameters
    ----------
    min_train_months : int
        Minimum history required before the first origin is emitted. Not part of the
        spec — a tunable warm-up choice.
    window_months : int, optional
        None (default) = expanding window. An integer = trailing sliding window of that
        many months, forwarded to train_frame.
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

        # truncate history at the origin through the single leakage boundary
        # (expanding by default; sliding if window_months is set)
        train_df = train_frame(nixtla_df, origin, window_months=window_months)

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

