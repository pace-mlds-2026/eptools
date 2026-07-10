import glob
import os
import platform
import string
import pandas as pd


_DATAFRAMES_CACHE: dict = {}


def _find_local_data_path():
    """Search for the DATA directory on Mac and Windows Google Drive desktop installs."""
    if platform.system() == "Darwin":
        matches = glob.glob(os.path.expanduser(
            "~/Library/CloudStorage/GoogleDrive-*/Shared drives/EmployerProject/DATA"
        ))
        if matches:
            return matches[0]

    elif platform.system() == "Windows":
        for letter in string.ascii_uppercase:
            candidate = os.path.join(f"{letter}:\\", "Shared drives", "EmployerProject", "DATA")
            if os.path.exists(candidate):
                return candidate

    return os.environ.get("EPTOOLS_DATA_PATH")


def _resolve_data_path(data_path=None) -> str:
    """Return the resolved DATA directory path for the current environment."""
    if data_path is not None:
        return data_path

    try:
        from google.colab import drive
        if not os.path.isdir("/content/drive/Shared drives"):
            drive.mount("/content/drive")
        return "/content/drive/Shared drives/EmployerProject/DATA"
    except ImportError:
        resolved = _find_local_data_path()
        if resolved is None:
            raise EnvironmentError(
                "Could not find the DATA directory automatically. "
                "Set the EPTOOLS_DATA_PATH environment variable, "
                "or pass data_path explicitly: "
                "load_dataframes(data_path='/path/to/DATA')"
            )
        return resolved


def _freeze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark each column's underlying numpy array as read-only.

    Protects the internal cache from accidental mutation within this module.
    Object-dtype columns (strings) are silently skipped as they do not
    support the writeable flag.
    """
    for col in df.columns:
        try:
            df[col].values.flags.writeable = False
        except (ValueError, AttributeError):
            pass
    return df


def _copy_cache(resolved: str) -> dict:
    """Return editable copies of the cached DataFrames."""
    return {key: df.copy() for key, df in _DATAFRAMES_CACHE[resolved].items()}


def load_dataframes(data_path=None) -> dict:
    """
    Load the raw project data files.

    Results are cached in memory after the first call. Subsequent calls with
    the same path return fresh editable copies without re-reading from disk.

    Auto-detects the environment (Colab, Mac, Windows). If auto-detection
    fails, set the EPTOOLS_DATA_PATH environment variable or pass data_path
    explicitly.

    Args:
        data_path: Optional path to the DATA directory. Overrides auto-detection.

    Returns a dict with keys:
        'sales'      -> chile_suzuki_historical_sales.csv as a DataFrame
        'dictionary' -> forecasting_data_dictionary.xlsx as a DataFrame
    """
    resolved = _resolve_data_path(data_path)

    if resolved in _DATAFRAMES_CACHE:
        return _copy_cache(resolved)

    sales_df      = pd.read_csv(os.path.join(resolved, "chile_suzuki_historical_sales.csv"))
    dictionary_df = pd.read_excel(os.path.join(resolved, "forecasting_data_dictionary.xlsx"))

    _DATAFRAMES_CACHE[resolved] = {
        "sales":      _freeze(sales_df),
        "dictionary": _freeze(dictionary_df),
    }
    return _copy_cache(resolved)



# these columns were found to contain either zero values or one value
REDUNDANT_COLUMNS = [
    "CPI_Housing_Utilities",
    "Car_Registrations",
    "Deposit_Interest_Rate",
    "Import_Prices",
    "Minimum_Wages",
    "COUNTRY_BRAND_CHANNEL",
    "Country",
    "Brand",
    "Channel",
    "REGION",
]

REQUIRED_COLUMNS = [
    "sku_id",   
    "month",
    "demand", 
    "collision_flag"
]

def get_collision_sales_df():
    """
    Load and clean the sales data, returning only collision rows.

      - Drops redundant columns (constant or entirely null)
      - Parses the Date column to a month-start datetime
      - Filters out non-collision rows
      - Adds sku_id, month, and demand convenience columns

    Returns:
        DataFrame with only collision sales, ready for modelling.
    """
    dfs = load_dataframes()
    sales = dfs["sales"]
    sales = sales.drop(columns=[c for c in REDUNDANT_COLUMNS if c in sales.columns])

    sales["sku_id"] = sales["ts_id"].astype(str)
    sales["month"] = pd.to_datetime(sales["Date"]).dt.to_period("M").dt.to_timestamp()
    
    #Edwin: Removed .fillna(0) so that we are not padding the data with 0 when it isn't needed.
    sales["demand"] = pd.to_numeric(sales["value"], errors="coerce")

    sales["collision_flag_clean"] = (
        sales["collision_flag"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    sales["is_collision"] = (
        sales["collision_flag_clean"].str.contains("COLLISION")
        & ~sales["collision_flag_clean"].str.contains("NON")
    )

    #Ediwn: scope on the SKU, not the row: the collision flag was only backfilled from
    # Jan 2024 onward, so filtering rows drops ~55% of collision SKUs' pre-2024
    # history. Any SKU EVER flagged collision keeps its FULL history.
    collision_skus = sales.groupby("sku_id")["is_collision"].any()
    collision_skus = collision_skus[collision_skus].index
    sales = sales[sales["sku_id"].isin(collision_skus)]
    
    # drop all but the required columns i.e. ["sku_id", "month", "demand", "collision_flag"]
    sales = sales[REQUIRED_COLUMNS]
    return sales


# get the dictionary and strip out the redundant column information
def get_collision_sales_dictionary():
    dfs = load_dataframes()
    dictionary = dfs['dictionary']
    return dictionary[~dictionary["column_name"].isin(REDUNDANT_COLUMNS)]


#Edwin: Newly added functions for correctly getting the right data slices.
def get_sku_active_windows(panel):
    """
    One row per SKU: first_seen, last_seen. This is the scoring universe --
    which months a SKU is genuinely in scope for -- as distinct from a
    model's own training window.
    """
    return panel.groupby("sku_id")["month"].agg(first_seen="min", last_seen="max").reset_index()


def build_full_panel(panel, active_windows):
    """
    Reindex each SKU onto its OWN [first_seen, last_seen] window, filling any
    gap with zero.

    Deliberately per-SKU, not the global calendar. Reindexing to the full
    dataset range (as get_bare_sku_df below used to) fabricates years of
    zero-demand rows for any SKU that launches partway through the dataset,
    which biases both training data and downstream WMAPE.
    """
    full_index = pd.concat([
        pd.DataFrame({
            "sku_id": row.sku_id,
            "month": pd.date_range(row.first_seen, row.last_seen, freq="MS"),
        })
        for row in active_windows.itertuples()
    ], ignore_index=True)
    full = full_index.merge(panel, on=["sku_id", "month"], how="left")
    full["demand"] = full["demand"].fillna(0)
    return full.sort_values(["sku_id", "month"]).reset_index(drop=True)


def load_collision_backtest_data():
    """One-call setup for the standard case. Equivalent to:
        collision_sales = get_collision_sales_df()
        windows = get_sku_active_windows(collision_sales)
        full_panel = build_full_panel(collision_sales, windows)
    Call those three directly if you need a custom scope (a SKU subset,
    a different date range, a test panel)."""
    collision_sales = get_collision_sales_df()
    windows = get_sku_active_windows(collision_sales)
    full_panel = build_full_panel(collision_sales, windows)
    return full_panel, windows


def get_moirai_wide_format_df():

    collision_sales = get_collision_sales_df()
    all_months = pd.date_range(
        collision_sales["month"].min(),
        collision_sales["month"].max(),
        freq="MS"
    )

    wide_df = (
        collision_sales
        .pivot(
            index="month",
            columns="sku_id",
            values="demand"
        )
        .reindex(all_months)
        .fillna(0)
        .sort_index()
    )

    return wide_df



def get_all_months():
    return pd.date_range('2021-01-01', '2026-04-01', freq="MS", name='date')



#Edwin: Rewritten to avoid a per-SKU query() loop to reduce conversion time. From 18 minutes to 19 seconds.
def rebuild_nixtla_df():
    """ create a nixtla format dataframe for all relevant skus and save into the data directory
    # this is the nixtla format needed for TimeGPT, TSB etc
    # Column      NameData              TypeDescription
    # unique_id   String or Number      distinct identifier for one time series
    # ds          Datestamp or Integer  the time index
    # y           Numeric               the actual historical/target value

    Uses get_collision_sales_df() as the single source of truth for SKU scope --
    not get_bodywork_skus(), which derives its own separate list from
    FAMILY_DESCRIPTION == "CARROCERIA".
    """
    collision_sales = get_collision_sales_df()

    windows = get_sku_active_windows(collision_sales)
    full_panel = build_full_panel(collision_sales, windows)

    nixtla = full_panel.rename(
        columns={'sku_id': 'unique_id', 'month': 'ds', 'demand': 'y'}
    )[['unique_id', 'ds', 'y']]

    out_path = os.path.join(_resolve_data_path(), 'API_SOURCES', 'API_SOURCE_nixtla.parquet')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nixtla.to_parquet(out_path, index=False)

    return nixtla


def to_nixtla_format(train_df):
    return train_df.rename(columns={"sku_id": "unique_id", "month": "ds", "demand": "y"})[
        ["unique_id", "ds", "y"]
    ]


def get_skus_by_segment(sku_segment: pd.DataFrame, segment_col: str, values) -> list:
    """Return sku_ids matching one or more values in a segment column. Works
    with ANY classification table -- sb_class, abc_class, subfamily, whatever
    gets added later -- since it's parameterized on the column, not hardcoded
    to Syntetos-Boylan specifically."""
    if isinstance(values, str):
        values = [values]
    return sku_segment.loc[sku_segment[segment_col].isin(values), "sku_id"].tolist()


def subset_panel(full_panel, active_windows, sku_ids):
    """Restrict to a specific SKU subset, reusing the exact shape run_backtest
    expects -- so the result plugs straight back in unchanged."""
    return (full_panel[full_panel["sku_id"].isin(sku_ids)].reset_index(drop=True),
            active_windows[active_windows["sku_id"].isin(sku_ids)].reset_index(drop=True))


collision_sales = get_collision_sales_df()
collision_sales.head()


#Edwin: Moved here.
# forecast protocol constants — fixed by the project spec. Moved here from
# eptools.modelling since fold generation now lives in this module.
LAG_MONTHS = 3    # months between the last known observation and the scored forecast month
HORIZON    = 18   # full forecast curve length in months
import numpy as np
from typing import Any, Callable, Dict, Optional

def active_skus_at(active_windows: pd.DataFrame, target_month: pd.Timestamp) -> set:
    """
    Which SKUs are in scope for scoring at target_month -- i.e. whose
    [first_seen, last_seen] window contains it. A SKU that hasn't launched
    yet, or was discontinued before this month, is excluded here rather than
    being defaulted to y_pred=0 and silently scored as if it were an active,
    correctly-forecast-as-zero SKU.
    """
    mask = (
        (active_windows["first_seen"] <= target_month)
        & (active_windows["last_seen"] >= target_month)
    )
    return set(active_windows.loc[mask, "sku_id"])

#Edwin: reworked from previous work.
def train_frame(full_panel: pd.DataFrame, origin: pd.Timestamp, window_months: int | None = None) -> pd.DataFrame:
    """
    THE leakage boundary. Given an origin, return only rows with ds <= origin.
    window_months=None gives an expanding window (default); an int gives a
    trailing window of that many months -- worth sweeping given the
    structural demand decline documented in data_profile.md.
    """
    window_start = full_panel["month"].min() if window_months is None \
        else origin - pd.DateOffset(months=window_months - 1)
        
    tf = full_panel.loc[(full_panel["month"] >= window_start) & (full_panel["month"] <= origin)].copy()
    assert tf["month"].max() <= origin, f"LEAKAGE: train_frame returned data beyond origin {origin.date()}"
    return tf


def expanding_window_folds(
    full_panel: pd.DataFrame,
    active_windows: pd.DataFrame,
    min_train_months: int = 12,
    lag: int = LAG_MONTHS,
    horizon: int = HORIZON,
    window_months: int | None = None,
):
    """
    Rolling-origin, expanding-window fold generator -- same fixed 3-month lag
    and 18-month horizon as before, but built on the corrected panel:
    train_df is dense within each SKU's own active window (no fabricated
    pre-launch/post-discontinuation zeros), and test_df is restricted HERE,
    at generation time, to SKUs genuinely active at target_date -- rather
    than being filtered later inside the scorer.
    """
    all_months = pd.date_range(full_panel["month"].min(), full_panel["month"].max(), freq="MS")
    first_origin_idx = min_train_months - 1
    last_origin_idx = len(all_months) - 1 - lag
    if last_origin_idx < first_origin_idx:
        raise ValueError(f"Not enough months ({len(all_months)}) for min_train_months={min_train_months} and lag={lag}.")

    for origin_idx in range(first_origin_idx, last_origin_idx + 1):
        origin = all_months[origin_idx]
        target_date = all_months[origin_idx + lag]

        train_df = train_frame(full_panel, origin, window_months=window_months)

        horizon_end_idx = min(origin_idx + horizon, len(all_months) - 1)
        horizon_dates = all_months[origin_idx + 1: horizon_end_idx + 1]

        scored_skus = active_skus_at(active_windows, target_date)
        test_df = full_panel.loc[
            (full_panel["month"] == target_date) & (full_panel["sku_id"].isin(scored_skus))
        ].copy()

        yield {"origin": origin, 
               "target_date": target_date, 
               "horizon_dates": horizon_dates,
               "train_df": train_df, 
               "test_df": test_df}


def validate_forecast_fn(forecast_fn, **kwargs):
    """Run forecast_fn against a tiny synthetic case and check the output
    isn't obviously broken -- wrong columns, non-numeric, or a forecast
    wildly out of scale (the exact TSB reset_index() bug). Call this once
    on your own model before trusting a full backtest run."""
    tiny_train = pd.DataFrame({
        "sku_id": ["_TEST_A"]*4 + ["_TEST_B"]*4,
        "month": list(pd.date_range("2021-01-01", periods=4, freq="MS"))*2,
        "demand": [10.0, 12.0, 9.0, 11.0, 100.0, 120.0, 90.0, 110.0],
    })
    result = forecast_fn(tiny_train, horizon=2, **kwargs)
    required = {"sku_id", "month", "y_pred"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if not pd.api.types.is_numeric_dtype(result["y_pred"]):
        raise ValueError("y_pred is non-numeric")
    max_actual = tiny_train["demand"].max()
    if result["y_pred"].max() > max_actual * 50:
        raise ValueError(f"y_pred max ({result['y_pred'].max():.1f}) is over 50x the largest "
                         f"training value ({max_actual}) -- check for a stray index column "
                         f"(see the TSB reset_index() bug).")
    print("Looks fine.")


def wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted mean absolute percentage error: sum|error| / sum|actual|.
    Returns nan if actuals sum to zero, rather than dividing by zero."""
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")

def pooled_wmape(results_df: pd.DataFrame, by: Optional[list] = None):
    """Volume-weighted pooled WMAPE from atomic sums.

    Σ|error| / Σ|actual| over the rows. With `by`, returns one pooled WMAPE per
    group (e.g. by='target_date' for a per-month curve, or by a joined segment
    column). Never averages per-fold ratios — always divides summed sums.
    """
    if by is None:
        denom = results_df["abs_actual_sum"].sum()
        return float(results_df["abs_error_sum"].sum() / denom) if denom > 0 else float("nan")

    g = results_df.groupby(by)
    out = (g["abs_error_sum"].sum() / g["abs_actual_sum"].sum()).rename("pooled_wmape")
    return out.reset_index()

def rolling_average_wmape(results_df, window=3, method="pooled"):
    r = results_df.sort_values("origin").reset_index(drop=True)
    if method == "pooled":
        return r["abs_error_sum"].rolling(window).sum() / r["abs_actual_sum"].rolling(window).sum()
    if method == "mean":
        return r["wmape"].rolling(window).mean()
    raise ValueError("method must be 'pooled' or 'mean'")


ForecastFn = Callable[..., pd.DataFrame]

#Edwin: 
"""removed the clip_negative params. clipping is model-specific behaviour
    (ARIMA/LSTM can emit negatives, Croston/TSB structurally cannot) and
    belongs inside whichever forecast_fn needs it, not as a harness default
    every model is forced through.
"""
def _joined_fold(fold, preds):
    """The per-SKU joined table for one fold: sku_id, demand, y_pred, plus
    which origin/target_date it came from. Excludes SKUs the model couldn't
    forecast (see n_skus_excluded_new). Shared by _score_fold (which reduces
    this to sums) and collect_predictions (which keeps it raw for plotting)."""
    target_date = fold["target_date"]
    target_preds = preds.loc[preds["month"] == target_date, ["sku_id", "y_pred"]].copy()
    merged = fold["test_df"][["sku_id", "demand"]].merge(target_preds, on="sku_id", how="left")
    missing_forecast = merged["y_pred"].isna()
    n_skus_excluded_new = int(missing_forecast.sum())
    merged = merged[~missing_forecast].copy()
    merged["origin"] = fold["origin"]
    merged["target_date"] = target_date
    return merged, n_skus_excluded_new
    
def _score_fold(fold: Dict[str, Any], preds: pd.DataFrame):
    """Turn one fold's forecast into a row of atomic sums.

    fold["test_df"] is already restricted to SKUs active at target_date. A SKU
    active at target but absent from preds launched inside the lag window --
    the model had no history to forecast it from -- so it's excluded from
    WMAPE and counted in n_skus_excluded_new, never silently zero-filled.
    """
    merged, n_skus_excluded_new = _joined_fold(fold, preds)
    err = merged["demand"] - merged["y_pred"]
    abs_actual_sum = float(merged["demand"].abs().sum())

    return {
        "origin": fold["origin"],
        "target_date": fold["target_date"],
        "n_skus": int(len(merged)),
        "n_skus_excluded_new": n_skus_excluded_new,
        "abs_error_sum": float(err.abs().sum()),
        "abs_actual_sum": abs_actual_sum,
        "error_sum": float(err.sum()),
        "wmape": wmape(merged["demand"], merged["y_pred"]),
    }
    
def _score_fold_segments(fold, preds, sku_segment, segment_col="sb_class"):
    """Same join as _score_fold, grouped by segment_col before summing.
    Tracks n_skus_excluded_new PER SEGMENT, since exclusion rates can differ
    meaningfully by segment."""
    target_date = fold["target_date"]
    target_preds = preds.loc[preds["month"] == target_date, ["sku_id", "y_pred"]].copy()

    merged = (fold["test_df"][["sku_id", "demand"]]
              .merge(target_preds, on="sku_id", how="left")
              .merge(sku_segment[["sku_id", segment_col]], on="sku_id", how="left"))

    missing_forecast = merged["y_pred"].isna()
    excluded_counts = merged.loc[missing_forecast].groupby(segment_col).size().rename("n_skus_excluded_new")

    merged = merged[~missing_forecast].copy()
    merged["abs_error"] = (merged["demand"] - merged["y_pred"]).abs()

    out = merged.groupby(segment_col).agg(
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("demand", lambda s: s.abs().sum()),
        n_skus=("sku_id", "count"),
    ).reset_index()
    out = out.merge(excluded_counts, on=segment_col, how="left")
    out["n_skus_excluded_new"] = out["n_skus_excluded_new"].fillna(0).astype(int)
    out["origin"] = fold["origin"]
    out["target_date"] = target_date
    return out


# #Edwin: a newer and cleaner version.
# def run_backtest(forecast_fn: ForecastFn, 
#                  full_panel: pd.DataFrame, 
#                  active_windows: pd.DataFrame,
#                  params: Optional[Dict[str, Any]] = None, 
#                  min_train_months: int = 12, 
#                  on_fold_complete: Optional[Callable[[int, float], bool]] = None,
#                  collect_predictions=False,
#                  verbose: bool = False,) -> pd.DataFrame:
#     """Run the fixed rolling-origin backtest for one model configuration.

#     forecast_fn : forecast_fn(train_df, horizon=len(horizon_dates), **params)
#     full_panel, active_windows : the corrected panel and SKU windows, e.g.
#         from get_collision_sales_df() -> get_sku_active_windows() -> build_full_panel().
#     min_train_months : warm-up history before the first origin. Keep identical
#         across models being compared.
#     on_fold_complete : callable(fold_idx, running_pooled_wmape) -> bool, optional.
#         Called after every fold. Return True to stop early. This is how
#         Optuna pruning, a manual early-stop rule, or nothing at all hooks
#         into per-fold progress -- this function never imports or references
#         Optuna directly.
#     collect_predictions : bool, default False
#         If True, also returns a second DataFrame with every SKU-level
#         (actual, forecast) pair across the backtest -- for plotting one
#         SKU's history. Adds negligible cost (a merge already available from
#         the same forecast_fn call) since forecast_fn itself only runs once
#         per fold either way.

#     Returns
#     -------
#     pd.DataFrame
#         One row per fold: origin, target_date, n_skus, abs_error_sum,
#         abs_actual_sum, error_sum, wmape. Sorted by origin.
#     """
#     params = params or {}
#     rows = []
#     prediction_rows = [] if collect_predictions else None
#     run_abs_err = 0.0
#     run_abs_act = 0.0

#     fold_iter = enumerate(expanding_window_folds(
#         full_panel, active_windows, min_train_months=min_train_months
#     ))
#     for fold_idx, fold in fold_iter:
#         preds = forecast_fn(fold["train_df"], horizon=len(fold["horizon_dates"]), **params)
#         row = _score_fold(fold, preds)
#         rows.append(row)
        
#         if collect_predictions:
#             merged, _ = _joined_fold(fold, preds)  # cheap: reuses the SAME preds, no re-fit
#             merged["abs_error"] = (merged["demand"] - merged["y_pred"]).abs()
#             prediction_rows.append(merged)

#         run_abs_err += row["abs_error_sum"]
#         run_abs_act += row["abs_actual_sum"]

#         if verbose:
#             print(
#                 f"fold {fold_idx + 1:>2}  origin={row['origin']:%Y-%m}  "
#                 f"target={row['target_date']:%Y-%m}  wmape={row['wmape']:.4f}"
#             )

#         if on_fold_complete is not None:
#             running_pooled = run_abs_err / run_abs_act if run_abs_act > 0 else float("nan")
#             if on_fold_complete(fold_idx, running_pooled):
#                 break
    
#     results_df = pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)

#     if collect_predictions:
#         predictions_df = (pd.concat(prediction_rows, ignore_index=True)
#                           .sort_values(["sku_id", "origin"]).reset_index(drop=True))
#         return results_df, predictions_df
#     return results_df


#Edwin: 10th July - merged segmentation into run_backtest
def run_backtest(forecast_fn, full_panel, 
                active_windows, params=None, min_train_months=12,
                on_fold_complete=None, collect_predictions=False, verbose=False,
                sku_segment=None, segment_col="sb_class"):
    """
    If sku_segment is None (default): one row per fold, pooled across all
    scored SKUs -- exactly what this function always returned.

    If sku_segment is given: one row per (fold, segment) instead. Calling
    pooled_wmape(results) with no `by` still gives the correct overall
    pooled number in this case, since segments are disjoint and their sums
    reconstruct the total exactly (see the share-weighted identity). There
    is deliberately no separate 'ALL' row mixed in alongside the segment
    rows -- that would double-count every SKU-month's error.
    """
    params = params or {}
    rows = []
    prediction_rows = [] if collect_predictions else None
    run_abs_err = run_abs_act = 0.0

    for fold_idx, fold in enumerate(expanding_window_folds(full_panel, active_windows, min_train_months=min_train_months)):
        preds = forecast_fn(fold["train_df"], horizon=len(fold["horizon_dates"]), **params)

        if sku_segment is None:
            row = _score_fold(fold, preds)
            rows.append(row)
            fold_abs_err, fold_abs_act = row["abs_error_sum"], row["abs_actual_sum"]
        else:
            seg_df = _score_fold_segments(fold, preds, sku_segment, segment_col=segment_col)
            rows.extend(seg_df.to_dict("records"))
            fold_abs_err, fold_abs_act = seg_df["abs_error_sum"].sum(), seg_df["abs_actual_sum"].sum()

        if collect_predictions:
            merged, _ = _joined_fold(fold, preds)
            merged["abs_error"] = (merged["demand"] - merged["y_pred"]).abs()
            if sku_segment is not None:
                merged = merged.merge(sku_segment, on="sku_id", how="left")
            prediction_rows.append(merged)

        run_abs_err += fold_abs_err
        run_abs_act += fold_abs_act
        if verbose:
            wmape_this_fold = fold_abs_err / fold_abs_act if fold_abs_act > 0 else float("nan")
            print(f"fold {fold_idx+1:>2}  origin={fold['origin']:%Y-%m}  target={fold['target_date']:%Y-%m}  wmape={wmape_this_fold:.4f}")
        if on_fold_complete is not None:
            running_pooled = run_abs_err / run_abs_act if run_abs_act > 0 else float("nan")
            if on_fold_complete(fold_idx, running_pooled):
                break

    results_df = pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)
    if collect_predictions:
        predictions_df = pd.concat(prediction_rows, ignore_index=True).sort_values(["sku_id","origin"]).reset_index(drop=True)
        return results_df, predictions_df
    return results_df


def summarise_backtest(results_df, window=3, verbose=False):
    """The standard headline numbers for any backtest run, so every model
    comparison in this project reports the same things the same way."""
    total_pooled = pooled_wmape(results_df)
    rolling = rolling_average_wmape(results_df, window=window, method="pooled")
    latest_rolling = rolling.iloc[-1]
    latest_origins = results_df.sort_values("origin")["target_date"].iloc[-window:].dt.strftime("%Y-%m").tolist()

    if verbose:
        print(f"folds: {len(results_df)}")
    print(f"total pooled WMAPE (whole backtest): {total_pooled:.4f}")
    print(f"latest {window}-month rolling pooled WMAPE (covering {latest_origins}): {latest_rolling:.4f}")
    if "n_skus_excluded_new" in results_df.columns and results_df["n_skus_excluded_new"].sum() > 0:
        print(f"SKUs excluded (launched mid-lag-window): {results_df['n_skus_excluded_new'].sum()}")
    return {"total_pooled": total_pooled, "latest_rolling": latest_rolling}


def quick_backtest(forecast_fn, params=None, min_train_months=12, verbose=False):
    """First-look convenience: load data, run, summarize, in one call.
    For anything beyond a first look -- segmented results, a custom panel,
    a hyperparameter sweep -- use run_backtest directly, since this wrapper
    hides exactly the pieces you'd need to customize."""
    full_panel, windows = load_collision_backtest_data()
    results = run_backtest(forecast_fn, full_panel, windows, params=params,
                            min_train_months=min_train_months, verbose=verbose)
    summarise_backtest(results)
    return results


import matplotlib.pyplot as plt
def plot_sku_forecast_history(predictions_df, sku_id, full_panel=None, ax=None):
    """... same docstring, plus:
    ax : optional matplotlib Axes to draw into (e.g. one cell of a grid).
    If None (default), creates and shows its own figure -- unchanged
    standalone behaviour."""
    d = predictions_df[predictions_df["sku_id"] == sku_id].sort_values("target_date")
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 3.3))
    if full_panel is not None:
        hist = full_panel[full_panel["sku_id"] == sku_id]
        ax.plot(hist["month"], hist["demand"], color="#ccc", linewidth=1, label="full history")
    ax.plot(d["target_date"], d["demand"], marker="o", color="#2a78d6", label="actual", markersize=4)
    ax.plot(d["target_date"], d["y_pred"], marker="s", linestyle="--", color="#e07b39", label="forecast", markersize=4)
    ax.set_title(sku_id, fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    if standalone:
        plt.tight_layout()
        plt.show()
        
def pick_example_skus(full_panel, sku_segment=None, segment_col="sb_class", n_per_group=1, how="median"):
    """Pick representative example SKUs, one or more per group. If sku_segment
    is given, groups by that classification. If not, falls back to volume
    terciles computed directly from full_panel -- a sensible default on its
    own, since volume tier is the same commercial-priority axis as ABC tier,
    and needs no classification step to compute.

    how: "median" picks the SKU closest to the group's median volume (a
    typical example); "max" picks the highest-volume SKU (a flagship example)."""
    total_demand = full_panel.groupby("sku_id")["demand"].sum().rename("total_demand").reset_index()
    if sku_segment is not None:
        grouped = total_demand.merge(sku_segment[["sku_id", segment_col]], on="sku_id", how="left")
        group_col = segment_col
    else:
        grouped = total_demand.copy()
        grouped["volume_tier"] = pd.qcut(grouped["total_demand"].rank(method="first"),
                                          q=3, labels=["Low", "Medium", "High"])
        group_col = "volume_tier"

    examples = {}
    for group, g in grouped.groupby(group_col, observed=True):
        if len(g) == 0:
            continue
        if how == "median":
            target = g["total_demand"].median()
            pick = g.iloc[(g["total_demand"] - target).abs().argsort()[:n_per_group]]
        else:
            pick = g.nlargest(n_per_group, "total_demand")
        examples[group] = pick["sku_id"].tolist()
    return examples


def plot_example_skus(predictions_df, full_panel, sku_segment=None, segment_col="sb_class",
                       n_per_group=1, how="median"):
    """Plot plot_sku_forecast_history for a representative SKU per group --
    S-B class if sku_segment is given, volume tier otherwise. No changes
    needed to plot_sku_forecast_history itself."""
    examples = pick_example_skus(full_panel, sku_segment, segment_col, n_per_group, how)
    for group, sku_ids in examples.items():
        for sku_id in sku_ids:
            print(f"--- {group}: {sku_id} ---")
            plot_sku_forecast_history(predictions_df, sku_id, full_panel=full_panel)
            
def plot_example_skus_grid(predictions_df, full_panel, sku_segment=None, segment_col="sb_class",
                            n_per_group=2, how="median", figsize_per_cell=(4.2, 2.8)):
    """
    Grid version of plot_example_skus: one row per group (S-B class, or
    volume tier if sku_segment is None), up to n_per_group columns, all in
    one figure. If a group has fewer than n_per_group SKUs, the remaining
    cells in that row are left blank rather than erroring.
    """
    examples = pick_example_skus(full_panel, sku_segment, segment_col, n_per_group, how)
    groups = list(examples.keys())
    n_rows = len(groups)
    n_cols = max(len(v) for v in examples.values())

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(figsize_per_cell[0]*n_cols, figsize_per_cell[1]*n_rows),
                              squeeze=False)

    for row_i, group in enumerate(groups):
        sku_ids = examples[group]
        for col_i in range(n_cols):
            ax = axes[row_i][col_i]
            if col_i < len(sku_ids):
                plot_sku_forecast_history(predictions_df, sku_ids[col_i], full_panel=full_panel, ax=ax)
            else:
                ax.axis("off")
        axes[row_i][0].set_ylabel(str(group), fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_sku_horizon_snapshot(forecast_fn, full_panel, sku_id, origin, params=None, horizon=18):
    """One origin's full forecast curve for one SKU, overlaid on its actual
    history -- shows how the forecast degrades further from the origin,
    which plot_sku_forecast_history doesn't (that one only shows the single
    lag-scored point per fold, not the whole curve)."""
    params = params or {}
    tf = train_frame(full_panel, origin)
    preds = forecast_fn(tf, horizon=horizon, **params)
    preds_sku = preds[preds["sku_id"] == sku_id].sort_values("month")

    fig, ax = plt.subplots(figsize=(11, 4))
    hist = full_panel[(full_panel["sku_id"] == sku_id) & (full_panel["month"] <= origin)]
    future = full_panel[(full_panel["sku_id"] == sku_id) & (full_panel["month"] > origin)]
    ax.plot(hist["month"], hist["demand"], marker="o", color="#2a78d6", label="known history")
    ax.plot(future["month"], future["demand"], marker="o", color="#2a78d6", alpha=0.3, label="actual (not yet known at origin)")
    ax.plot(preds_sku["month"], preds_sku["y_pred"], marker="s", linestyle="--", color="#e07b39", label="forecast curve")
    ax.axvline(origin, color="black", linestyle=":", label="origin")
    ax.set_title(f"{sku_id}: full horizon forecast from origin={origin.date()}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def illustrate_folds(forecast_fn, full_panel, active_windows, sku_ids, origins,
                      params=None, lag=LAG_MONTHS, horizon=6, show_computations=True,
                      ylim=None, save_prefix=None):
    """
    Walk through a handful of specific origins, one plot per origin, showing
    the named SKU(s)' known history, forecast curve, and the exact error at
    the scored (lag) target month. This is the illustrative "here's how the
    harness works, and how multiple SKUs pool into one WMAPE" view -- for
    the full rolling series across every origin, use run_backtest directly.

    sku_ids : a single sku_id, or a list of 2-4 for a comparison view within
        each fold (more than ~4 gets visually unreadable).
    origins : specific origin months to illustrate, e.g.
        ["2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01"]
    show_computations : if True (default), adds the right-hand text panel
        with each SKU's actual/forecast/error and the fold's WMAPE. Set
        False for a cleaner, presentation-only version of the same plot.
    """
    params = params or {}
    if isinstance(sku_ids, str):
        sku_ids = [sku_ids]

    colors = ["#2a78d6", "#e07b39", "#4caf50", "#9c27b0"]
    origins = sorted(pd.Timestamp(o) for o in origins)

    if ylim is None:
        relevant = full_panel[full_panel["sku_id"].isin(sku_ids)]["demand"]
        ylim = (0, relevant.max() * 1.3)

    for i, origin in enumerate(origins, start=1):
        target_date = origin + pd.DateOffset(months=lag)
        all_months = pd.date_range(full_panel["month"].min(), full_panel["month"].max(), freq="MS")
        origin_idx = all_months.get_loc(origin)
        horizon_end_idx = min(origin_idx + horizon, len(all_months) - 1)
        horizon_dates = all_months[origin_idx + 1: horizon_end_idx + 1]

        train_df = train_frame(full_panel, origin)
        scored_skus = active_skus_at(active_windows, target_date)
        test_df = full_panel.loc[
            (full_panel["month"] == target_date) & (full_panel["sku_id"].isin(scored_skus))
        ].copy()
        fold = {"origin": origin, "target_date": target_date, "horizon_dates": horizon_dates,
                "train_df": train_df, "test_df": test_df}

        preds = forecast_fn(train_df, horizon=len(horizon_dates), **params)
        row = _score_fold(fold, preds)
        merged, _ = _joined_fold(fold, preds)

        if show_computations:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [2.3, 1]})
            ax, ax2 = axes
        else:
            fig, ax = plt.subplots(figsize=(11, 4.3))

        for sku, color in zip(sku_ids, colors):
            hist = full_panel[(full_panel["sku_id"] == sku) & (full_panel["month"] <= origin)]
            ax.plot(hist["month"], hist["demand"], marker="o", color=color, label=f"{sku} actual (known)")

            fc_sku = preds[preds["sku_id"] == sku]
            ax.plot(fc_sku["month"], fc_sku["y_pred"], marker="s", linestyle="--", color=color,
                    alpha=0.6, label=f"{sku} forecast")

            act_row = full_panel[(full_panel["sku_id"] == sku) & (full_panel["month"] == target_date)]
            fc_row = fc_sku[fc_sku["month"] == target_date]
            if len(act_row) and len(fc_row):
                a, f = act_row["demand"].values[0], fc_row["y_pred"].values[0]
                ax.plot([target_date, target_date], [a, f], color=color, linewidth=3, zorder=5)
                ax.scatter([target_date], [a], color=color, s=70, zorder=6, edgecolor="black")
                ax.scatter([target_date], [f], color=color, marker="s", s=70, zorder=6, edgecolor="black")
                ax.annotate(f"|err|={abs(a-f):.2f}", (target_date, max(a, f) + ylim[1]*0.03),
                            ha="center", fontsize=8, color=color)

        ax.axvline(origin, color="black", linestyle=":", linewidth=1.2)
        ax.axvline(target_date, color="gray", linestyle=":", linewidth=1.2)
        ax.set_ylim(*ylim)
        ax.set_title(f"Fold {i}: origin={origin.strftime('%b %Y')}  ->  scored at {target_date.strftime('%b %Y')}")
        ax.legend(fontsize=7, loc="lower left")

        if show_computations:
            ax2.axis("off")
            lines = [f"Target: {target_date.strftime('%b %Y')}", ""]
            for r in merged[merged["sku_id"].isin(sku_ids)].itertuples():
                lines.append(f"{r.sku_id}: actual={r.demand:.0f}  forecast={r.y_pred:.2f}  "
                             f"|err|={abs(r.demand - r.y_pred):.2f}")
            lines += ["", f"sum |errors| (all scored SKUs) = {row['abs_error_sum']:.3f}",
                      f"sum |actuals| (all scored SKUs) = {row['abs_actual_sum']:.3f}", "",
                      f"WMAPE = {row['wmape']:.4f}"]
            ax2.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=10.5, family="monospace")

        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_fold{i}_{origin.strftime('%b').lower()}.png", bbox_inches="tight")
        plt.show()
        print(f"  Fold {i}: origin={origin.date()}  target={target_date.date()}  WMAPE={row['wmape']:.4f}")


def classify_sb(panel: pd.DataFrame) -> pd.DataFrame:
    """Syntetos-Boylan classification. Pass full_panel for a retrospective,
    reporting-only label. Pass train_frame(full_panel, origin) for a
    point-in-time label safe to use as a model feature or routing signal --
    never use the retrospective version for routing, only for slicing results."""
    is_demand = panel["demand"] > 0
    stats = panel.assign(is_demand=is_demand).groupby("sku_id").agg(
        n_periods=("month", "count"),
        n_demand_periods=("is_demand", "sum"),
        mean_nonzero=("demand", lambda s: s[s > 0].mean()),
        std_nonzero=("demand", lambda s: s[s > 0].std()),
    ).reset_index()

    stats["adi"] = stats["n_periods"] / stats["n_demand_periods"].replace(0, np.nan)
    stats["cv2"] = (stats["std_nonzero"] / stats["mean_nonzero"]) ** 2

    def label(row):
        if row["n_demand_periods"] == 0:
            return "NoDemand"
        smooth_adi, smooth_cv2 = row["adi"] < 1.32, row["cv2"] < 0.49
        if smooth_adi and smooth_cv2: return "Smooth"
        if smooth_adi: return "Erratic"
        if smooth_cv2: return "Intermittent"
        return "Lumpy"

    stats["sb_class"] = stats.apply(label, axis=1)
    return stats[["sku_id", "adi", "cv2", "sb_class"]]


# def _score_fold_segments(fold, preds, sku_segment: pd.DataFrame, segment_col: str = "sb_class") -> pd.DataFrame:
#     """Same as _score_fold, but returns one row PER SEGMENT per fold instead
#     of one row for the whole fold. sku_segment: DataFrame[sku_id, segment_col]."""
#     target_date = fold["target_date"]
#     target_preds = preds.loc[preds["month"] == target_date, ["sku_id", "y_pred"]].copy()

#     merged = fold["test_df"][["sku_id", "demand"]].merge(target_preds, on="sku_id", how="left")
#     missing_forecast = merged["y_pred"].isna()
#     merged = merged[~missing_forecast].merge(sku_segment, on="sku_id", how="left")
#     merged["abs_error"] = (merged["demand"] - merged["y_pred"]).abs()

#     out = merged.groupby(segment_col).agg(
#         abs_error_sum=("abs_error", "sum"),
#         abs_actual_sum=("demand", lambda s: s.abs().sum()),
#         n_skus=("sku_id", "count"),
#     ).reset_index()
#     out["origin"], out["target_date"] = fold["origin"], target_date
#     return out


# def run_backtest_segmented(forecast_fn, full_panel, active_windows, sku_segment,
#                           segment_col="sb_class", params=None, min_train_months=12) -> pd.DataFrame:
#     """Same fold generator as run_backtest, reused unchanged -- only scoring differs."""
#     params = params or {}
#     all_segs = []
#     for fold in expanding_window_folds(full_panel, active_windows, min_train_months=min_train_months):
#         preds = forecast_fn(fold["train_df"], horizon=len(fold["horizon_dates"]), **params)
#         all_segs.append(_score_fold_segments(fold, preds, sku_segment, segment_col=segment_col))
#     return pd.concat(all_segs, ignore_index=True)


# # NOTEBOOK_ONLY
# tsb_segmented = run_backtest_segmented(tsb_forecast_fn, full_panel, windows, sb_lookup, min_train_months=12)
# pooled_wmape(tsb_segmented, by="sb_class")            # one number per segment, whole backtest
