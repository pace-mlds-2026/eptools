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



#Edwin: I think this can be remove as it isn't used anymore.
def get_bare_sku_df(sku_code, format=None, no_warnings=False, _sales=None):
    """ return a dataframe with a dateTime index spanning THIS SKU's own active
        window (first sale to last sale), with the demand for that period.
        zero values will be filled for any gap inside that window

        _sales: optionally pass a pre-loaded sales DataFrame to avoid repeated copying when
                calling this function in a loop (load_dataframes() copies the full DataFrame
                on every call, so passing it in pays that cost once instead of per-SKU)
    """

    sales = _sales if _sales is not None else load_dataframes()['sales']

    sku_rows = (
        sales[['Date', 'value', 'ts_id']]
        .query('ts_id == @sku_code')
        .assign(Date=lambda df: pd.to_datetime(df['Date']))
    )

    if len(sku_rows) == 0:
        raise ValueError(f'SKU not found: {sku_code}')

    # reindex onto THIS SKU's own [first_seen, last_seen] window, not the
    # global calendar -- reindexing to the full dataset range fabricates
    # zero-demand rows before launch / after discontinuation, which biases
    # training data and any classification (e.g. Syntetos-Boylan) computed
    # from it.
    sku_months = pd.date_range(sku_rows['Date'].min(), sku_rows['Date'].max(), freq="MS")

    sku_sales = (
        sku_rows
        .set_index('Date')[['value']]
        .reindex(sku_months, fill_value=0)
        .rename(columns={'value': 'y'})
    )

    if not no_warnings:
        missing_months = (sku_sales['y'] == 0).sum()
        if missing_months > 0:
            print(f"WARNING: {missing_months} months were missing data and defaulted to 0")

    return sku_sales


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
def _score_fold(fold: Dict[str, Any], preds: pd.DataFrame) -> Dict[str, Any]:
    """Turn one fold's forecast into a row of atomic sums.

    fold["test_df"] is already restricted to SKUs active at target_date. A SKU
    active at target but absent from preds launched inside the lag window --
    the model had no history to forecast it from -- so it's excluded from
    WMAPE and counted in n_skus_excluded_new, never silently zero-filled.
    """
    target_date = fold["target_date"]
    target_preds = preds.loc[preds["month"] == target_date, ["sku_id", "y_pred"]].copy()

    merged = fold["test_df"][["sku_id", "demand"]].merge(target_preds, on="sku_id", how="left")

    missing_forecast = merged["y_pred"].isna()
    n_skus_excluded_new = int(missing_forecast.sum())
    merged = merged[~missing_forecast]

    err = merged["demand"] - merged["y_pred"]
    abs_actual_sum = float(merged["demand"].abs().sum())

    return {
        "origin": fold["origin"],
        "target_date": target_date,
        "n_skus": int(len(merged)),
        "n_skus_excluded_new": n_skus_excluded_new,
        "abs_error_sum": float(err.abs().sum()),
        "abs_actual_sum": abs_actual_sum,
        "error_sum": float(err.sum()),
        "wmape": wmape(merged["demand"], merged["y_pred"]),
    }


#Edwin: a newer and cleaner version.
def run_backtest(forecast_fn: ForecastFn, 
                 full_panel: pd.DataFrame, 
                 active_windows: pd.DataFrame,
                 params: Optional[Dict[str, Any]] = None, 
                 min_train_months: int = 12, 
                 on_fold_complete: Optional[Callable[[int, float], bool]] = None,
                 verbose: bool = False,) -> pd.DataFrame:
    """Run the fixed rolling-origin backtest for one model configuration.

    forecast_fn : forecast_fn(train_df, horizon=len(horizon_dates), **params)
    full_panel, active_windows : the corrected panel and SKU windows, e.g.
        from get_collision_sales_df() -> get_sku_active_windows() -> build_full_panel().
    min_train_months : warm-up history before the first origin. Keep identical
        across models being compared.
    on_fold_complete : callable(fold_idx, running_pooled_wmape) -> bool, optional.
        Called after every fold. Return True to stop early. This is how
        Optuna pruning, a manual early-stop rule, or nothing at all hooks
        into per-fold progress -- this function never imports or references
        Optuna directly.

    Returns
    -------
    pd.DataFrame
        One row per fold: origin, target_date, n_skus, abs_error_sum,
        abs_actual_sum, error_sum, wmape. Sorted by origin.
    """
    params = params or {}
    rows = []
    run_abs_err = 0.0
    run_abs_act = 0.0

    fold_iter = enumerate(expanding_window_folds(
        full_panel, active_windows, min_train_months=min_train_months
    ))
    for fold_idx, fold in fold_iter:
        preds = forecast_fn(fold["train_df"], horizon=len(fold["horizon_dates"]), **params)
        row = _score_fold(fold, preds)
        rows.append(row)

        run_abs_err += row["abs_error_sum"]
        run_abs_act += row["abs_actual_sum"]

        if verbose:
            print(
                f"fold {fold_idx + 1:>2}  origin={row['origin']:%Y-%m}  "
                f"target={row['target_date']:%Y-%m}  wmape={row['wmape']:.4f}"
            )

        if on_fold_complete is not None:
            running_pooled = run_abs_err / run_abs_act if run_abs_act > 0 else float("nan")
            if on_fold_complete(fold_idx, running_pooled):
                break

    return pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)


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


def _score_fold_segments(fold, preds, sku_segment: pd.DataFrame, segment_col: str = "sb_class") -> pd.DataFrame:
    """Same as _score_fold, but returns one row PER SEGMENT per fold instead
    of one row for the whole fold. sku_segment: DataFrame[sku_id, segment_col]."""
    target_date = fold["target_date"]
    target_preds = preds.loc[preds["month"] == target_date, ["sku_id", "y_pred"]].copy()

    merged = fold["test_df"][["sku_id", "demand"]].merge(target_preds, on="sku_id", how="left")
    missing_forecast = merged["y_pred"].isna()
    merged = merged[~missing_forecast].merge(sku_segment, on="sku_id", how="left")
    merged["abs_error"] = (merged["demand"] - merged["y_pred"]).abs()

    out = merged.groupby(segment_col).agg(
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("demand", lambda s: s.abs().sum()),
        n_skus=("sku_id", "count"),
    ).reset_index()
    out["origin"], out["target_date"] = fold["origin"], target_date
    return out


def run_backtest_segmented(forecast_fn, full_panel, active_windows, sku_segment,
                          segment_col="sb_class", params=None, min_train_months=12) -> pd.DataFrame:
    """Same fold generator as run_backtest, reused unchanged -- only scoring differs."""
    params = params or {}
    all_segs = []
    for fold in expanding_window_folds(full_panel, active_windows, min_train_months=min_train_months):
        preds = forecast_fn(fold["train_df"], horizon=len(fold["horizon_dates"]), **params)
        all_segs.append(_score_fold_segments(fold, preds, sku_segment, segment_col=segment_col))
    return pd.concat(all_segs, ignore_index=True)


def to_nixtla_format(train_df):
    return train_df.rename(columns={"sku_id": "unique_id", "month": "ds", "demand": "y"})[
        ["unique_id", "ds", "y"]
    ]


from statsforecast import StatsForecast
from statsforecast.models import TSB

def tsb_forecast_fn(train_df, horizon=18, alpha_d=0.2, alpha_p=0.2):
    """TSB via statsforecast. Nixtla conversion happens here only --
    the harness never sees unique_id/ds/y."""
    nixtla_df = to_nixtla_format(train_df)
    sf = StatsForecast(models=[TSB(alpha_d=alpha_d, alpha_p=alpha_p)], freq="MS")
    raw = sf.forecast(df=nixtla_df, h=horizon)  # no reset_index() -- unique_id/ds are already columns

    pred_col = [c for c in raw.columns if c not in ("unique_id", "ds")][0]
    origin = train_df["month"].max()

    out = raw.rename(columns={"unique_id": "sku_id", "ds": "month", pred_col: "y_pred"})
    out["h"] = (
        (out["month"].dt.year - origin.year) * 12
        + (out["month"].dt.month - origin.month)
    )
    return out[["sku_id", "month", "h", "y_pred"]]
