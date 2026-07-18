import glob
import os
import platform
import string
import pandas as pd
import matplotlib.pyplot as plt


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



LAG_MONTHS = 3    # months between the last known observation and the scored forecast month
HORIZON    = 18   # full forecast curve length in months

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
# def get_sku_active_windows(panel):
#     """
#     One row per SKU: first_seen, last_seen. This is the scoring universe --
#     which months a SKU is genuinely in scope for -- as distinct from a
#     model's own training window.
#     """
#     return panel.groupby("sku_id")["month"].agg(first_seen="min", last_seen="max").reset_index()

def get_sku_active_windows(panel):
    """
    One row per SKU containing the first and final row observed in the
    supplied extract.

    Important:
    - first_seen is the first observed SKU-month in the extract. Inchcape
      describes this as the first observed invoiced sale, not necessarily
      the product introduction date.
    - last_seen is the final observed row in the extract, not necessarily
      a retirement or discontinuation date.

    In the current Chile Suzuki collision panel, every SKU continues through
    the dataset end, April 2026.
    """
    required = {"sku_id", "month"}
    missing = required - set(panel.columns)

    if missing:
        raise ValueError(
            f"panel is missing required columns: {missing}"
        )

    return panel.groupby("sku_id")["month"].agg(first_seen="min",last_seen="max",).reset_index()

    
# def build_full_panel(panel, active_windows):
#     """
#     Reindex each SKU onto its OWN [first_seen, last_seen] window, filling any
#     gap with zero.

#     Deliberately per-SKU, not the global calendar. Reindexing to the full
#     dataset range (as get_bare_sku_df below used to) fabricates years of
#     zero-demand rows for any SKU that launches partway through the dataset,
#     which biases both training data and downstream WMAPE.
#     """
#     full_index = pd.concat([
#         pd.DataFrame({
#             "sku_id": row.sku_id,
#             "month": pd.date_range(row.first_seen, row.last_seen, freq="MS"),
#         })
#         for row in active_windows.itertuples()
#     ], ignore_index=True)
#     full = full_index.merge(panel, on=["sku_id", "month"], how="left")
#     full["demand"] = full["demand"].fillna(0)
#     return full.sort_values(["sku_id", "month"]).reset_index(drop=True)

def build_full_panel(panel, active_windows, fill_internal_gaps=False):
    """
    Reindex each SKU onto its OWN [first_seen, last_seen] window, filling any
    gap with zero.

    Deliberately per-SKU, not the global calendar.
    
    Validate and, only when explicitly requested, complete each SKU's
    observed first-row-to-last-row monthly sequence.

    By default, this function raises an error if internal SKU-month rows
    are missing. It does not silently interpret missing records as
    zero invoiced sales.

    For the current Chile Suzuki collision data, no rows should be added.
    """
    required_panel = {"sku_id","month","demand"}
    
    missing_panel = required_panel - set(panel.columns)

    if missing_panel:
        raise ValueError(
            f"panel is missing required columns: "
            f"{missing_panel}"
        )

    required_windows = { "sku_id", "first_seen", "last_seen"}
    
    missing_windows = required_windows - set(active_windows.columns)
    

    if missing_windows:
        raise ValueError(
            f"active_windows is missing required columns: "
            f"{missing_windows}"
        )

    if panel.duplicated(["sku_id", "month"]).any():
        raise ValueError("panel contains duplicate SKU-month rows.")

    full_index = pd.concat(
        [
            pd.DataFrame(
                {
                    "sku_id": row.sku_id,
                    "month": pd.date_range(
                        row.first_seen,
                        row.last_seen,
                        freq="MS",
                    ),
                }
            )
            for row in active_windows.itertuples()
        ],
        ignore_index=True,
    )

    full = full_index.merge(
        panel,
        on=["sku_id", "month"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    added_mask = full["_merge"] == "left_only"
    n_rows_added = int(added_mask.sum())

    if n_rows_added > 0 and not fill_internal_gaps:
        examples = (
            full.loc[
                added_mask,
                ["sku_id", "month"],
            ]
            .head(20)
        )

        display(examples)

        raise ValueError(
            f"Found {n_rows_added:,} missing internal "
            f"SKU-month rows. These have not been filled "
            f"because fill_internal_gaps=False."
        )

    if n_rows_added > 0:
        full.loc[
            added_mask,
            "demand",
        ] = 0.0

        print(
            f"Explicitly filled {n_rows_added:,} "
            f"internal SKU-month gaps with zero."
        )

    full = full.drop(columns="_merge")

    if full["demand"].isna().any():
        raise ValueError(
            "The completed panel still contains missing "
            "demand values."
        )

    return full.sort_values(["sku_id", "month"]).reset_index(drop=True)

def skus_observed_by(
    active_windows: pd.DataFrame,
    month: pd.Timestamp,
) -> set:
    """
    SKUs whose first observed row is no later than month.

    This is based only on information contained in the supplied demand
    extract. It does not claim that first_seen is the product launch date.
    """
    month = (
        pd.Timestamp(month)
        .to_period("M")
        .to_timestamp()
    )

    return set(
        active_windows.loc[
            active_windows["first_seen"] <= month,
            "sku_id",
        ]
    )


def define_scoring_universe(active_windows: pd.DataFrame, origin: pd.Timestamp, target_date: pd.Timestamp,
                            scoring_policy: str,):
    """
    Define eligibility before examining model predictions.

    Policies
    --------
    known_at_origin:
        Score only SKUs that had appeared in the supplied demand history
        by the forecast origin.

    all_observed_by_target:
        Score every SKU that had appeared by the target month. SKUs first
        observed after the origin require an explicit cold-start rule.
    """
    known_at_origin = skus_observed_by(active_windows,origin,)

    observed_by_target = skus_observed_by(active_windows,target_date,)

    post_origin_first_sale_skus = observed_by_target - known_at_origin
    

    if scoring_policy == "known_at_origin":
        scored_skus = known_at_origin
    elif scoring_policy == "all_observed_by_target":
        scored_skus = observed_by_target
    else:
        raise ValueError(
            "scoring_policy must be either "
            "'known_at_origin' or "
            "'all_observed_by_target'."
        )

    return {
        "scored_skus": scored_skus,
        "known_at_origin": known_at_origin,
        "observed_by_target": observed_by_target,
        "post_origin_first_sale_skus": post_origin_first_sale_skus
    }


def load_collision_backtest_data():
    """One-call setup for the standard case. Equivalent to:
        collision_sales = get_collision_sales_df()
        windows = get_sku_active_windows(collision_sales)
        full_panel = build_full_panel(collision_sales, windows)
    Call those three directly if you need a custom scope (a SKU subset,
    a different date range, a test panel)."""
    collision_sales = get_collision_sales_df()
    windows = get_sku_active_windows(collision_sales)
    full_panel = build_full_panel(collision_sales, windows, fill_internal_gaps=False)
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


def make_fold(
    full_panel,
    active_windows,
    origin,
    lag=LAG_MONTHS,
    horizon=HORIZON,
    window_months=None,
    scoring_policy="known_at_origin",
):
    """Build one complete fold using the same eligibility rules as the backtest."""
    origin = pd.Timestamp(origin).to_period("M").to_timestamp()
    target_date = origin + pd.DateOffset(months=lag)
    dataset_end = full_panel["month"].max()

    if target_date > dataset_end:
        raise ValueError(
            f"Target {target_date:%Y-%m} is after dataset end {dataset_end:%Y-%m}."
        )

    horizon_end = min(origin + pd.DateOffset(months=horizon), dataset_end)
    horizon_dates = pd.date_range(origin + pd.DateOffset(months=1),horizon_end,freq="MS",)

    train_df = train_frame(full_panel, origin, window_months=window_months)

    universe = define_scoring_universe(active_windows,origin,target_date,scoring_policy=scoring_policy,)

    target_actuals = full_panel.loc[full_panel["month"] == target_date,["sku_id", "month", "demand"]]

    test_df = pd.DataFrame({"sku_id": sorted(universe["scored_skus"])}).merge(target_actuals, on="sku_id", how="left", validate="one_to_one")

    if test_df["demand"].isna().any():
        n_missing = int(test_df["demand"].isna().sum())
        raise ValueError(
            f"{n_missing:,} eligible SKUs have no actual row for {target_date:%Y-%m}."
        )

    return {
        "origin": origin,
        "target_date": target_date,
        "horizon_dates": horizon_dates,
        "train_df": train_df,
        "test_df": test_df,
        "scoring_policy": scoring_policy,
        "known_at_origin": universe["known_at_origin"],
        "observed_by_target": universe["observed_by_target"],
        "post_origin_first_sale_skus": universe["post_origin_first_sale_skus"],
    }
    
def _apply_scope(fold, scope_fn):
    """
    Narrow a fold to a declared SKU scope, computed from that fold's own
    train_df -- already sliced to <= origin by train_frame(), so this is
    point-in-time by construction. classify_sb's own docstring says exactly
    this: pass train_frame(full_panel, origin) for a routing-safe label,
    never the full-panel version. scope_fn is how that rule gets enforced
    automatically rather than relying on every model author remembering it.

    The SAME scope decides what forecast_fn is allowed to see AND what the
    scorer expects back -- one classification call per fold, not two, so
    "what the model was given" and "what the scorer demands" can't drift
    apart the way they would if each model wrapped its own filtering.
    """
    scoped_skus = scope_fn(fold["train_df"])
    fold = dict(fold)  # shallow copy -- don't mutate the shared fold dict
    fold["train_df"] = fold["train_df"][fold["train_df"]["sku_id"].isin(scoped_skus)]
    fold["known_at_origin"] = fold["known_at_origin"] & scoped_skus
    fold["post_origin_first_sale_skus"] = fold["post_origin_first_sale_skus"] & scoped_skus
    fold["test_df"] = fold["test_df"][fold["test_df"]["sku_id"].isin(scoped_skus)]
    return fold

# def expanding_window_folds(
#     full_panel: pd.DataFrame,
#     active_windows: pd.DataFrame,
#     min_train_months: int = 12,
#     lag: int = LAG_MONTHS,
#     horizon: int = HORIZON,
#     window_months: int | None = None,
# ):
#     """
#     Rolling-origin, expanding-window fold generator -- same fixed 3-month lag
#     and 18-month horizon as before, but built on the corrected panel:
#     train_df is dense within each SKU's own active window (no fabricated
#     pre-launch/post-discontinuation zeros), and test_df is restricted HERE,
#     at generation time, to SKUs genuinely active at target_date -- rather
#     than being filtered later inside the scorer.
#     """
#     all_months = pd.date_range(full_panel["month"].min(), full_panel["month"].max(), freq="MS")
#     first_origin_idx = min_train_months - 1
#     last_origin_idx = len(all_months) - 1 - lag
#     if last_origin_idx < first_origin_idx:
#         raise ValueError(f"Not enough months ({len(all_months)}) for min_train_months={min_train_months} and lag={lag}.")

#     for origin_idx in range(first_origin_idx, last_origin_idx + 1):
#         origin = all_months[origin_idx]
#         target_date = all_months[origin_idx + lag]

#         train_df = train_frame(full_panel, origin, window_months=window_months)

#         horizon_end_idx = min(origin_idx + horizon, len(all_months) - 1)
#         horizon_dates = all_months[origin_idx + 1: horizon_end_idx + 1]

#         scored_skus = active_skus_at(active_windows, target_date)
#         test_df = full_panel.loc[
#             (full_panel["month"] == target_date) & (full_panel["sku_id"].isin(scored_skus))
#         ].copy()

#         yield {"origin": origin, 
#                "target_date": target_date, 
#                "horizon_dates": horizon_dates,
#                "train_df": train_df, 
#                "test_df": test_df}

def expanding_window_folds(
    full_panel: pd.DataFrame,
    active_windows: pd.DataFrame,
    min_train_months: int = 12,
    lag: int = LAG_MONTHS,
    horizon: int = HORIZON,
    window_months: int | None = None,
    scoring_policy: str = "known_at_origin",
):
    """
    Rolling-origin fold generator.

    Eligibility is defined before the forecasting model is run.

    scoring_policy='known_at_origin':
        Score SKUs observed by the forecast origin.

    scoring_policy='all_observed_by_target':
        Also score SKUs first observed between origin and target. These
        require an explicit cold-start forecast rule in the scorer.
    """
    all_months = pd.date_range(
        full_panel["month"].min(),
        full_panel["month"].max(),
        freq="MS",
    )

    first_origin_idx = min_train_months - 1
    last_origin_idx = len(all_months) - lag - 1

    if last_origin_idx < first_origin_idx:
        raise ValueError(
            f"Not enough history for min_train_months={min_train_months} and lag={lag}."
        )

    for origin in all_months[first_origin_idx:last_origin_idx + 1]:
        yield make_fold(
            full_panel,
            active_windows,
            origin,
            lag=lag,
            horizon=horizon,
            window_months=window_months,
            scoring_policy=scoring_policy,
        )


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
# def _joined_fold(fold, preds):
#     """The per-SKU joined table for one fold: sku_id, demand, y_pred, plus
#     which origin/target_date it came from. Excludes SKUs the model couldn't
#     forecast (see n_skus_excluded_new). Shared by _score_fold (which reduces
#     this to sums) and collect_predictions (which keeps it raw for plotting)."""
#     target_date = fold["target_date"]
#     target_preds = preds.loc[preds["month"] == target_date, ["sku_id", "y_pred"]].copy()
#     merged = fold["test_df"][["sku_id", "demand"]].merge(target_preds, on="sku_id", how="left")
#     missing_forecast = merged["y_pred"].isna()
#     n_skus_excluded_new = int(missing_forecast.sum())
#     merged = merged[~missing_forecast].copy()
#     merged["origin"] = fold["origin"]
#     merged["target_date"] = target_date
#     return merged, n_skus_excluded_new

def _joined_fold(fold,preds,cold_start_strategy="error"):
    """
    Join one fold's eligible actuals to its target predictions.

    Missing predictions for SKUs known at the forecast origin are always
    errors.

    For post-origin first-sale SKUs:
    - cold_start_strategy='zero' assigns an explicit zero forecast;
    - cold_start_strategy='error' raises an error.

    No prediction is silently dropped.
    """
    required = {
        "sku_id",
        "month",
        "y_pred",
    }
    missing_columns = required - set(
        preds.columns
    )

    if missing_columns:
        raise ValueError(
            f"Forecast output is missing columns: "
            f"{missing_columns}"
        )

    target_date = fold["target_date"]

    target_preds = (
        preds.loc[
            preds["month"] == target_date,
            [
                "sku_id",
                "y_pred",
            ],
        ]
        .copy()
    )

    duplicate_mask = (
        target_preds["sku_id"]
        .duplicated(keep=False)
    )

    if duplicate_mask.any():
        examples = (
            target_preds.loc[
                duplicate_mask
            ]
            .sort_values("sku_id")
            .head(20)
        )

        display(examples)

        raise ValueError(
            f"Found {duplicate_mask.sum():,} "
            f"duplicate prediction rows for "
            f"{target_date:%Y-%m}."
        )

    invalid_forecast_mask = (
        target_preds["y_pred"].isna()
        | ~np.isfinite(
            target_preds["y_pred"]
        )
    )

    if invalid_forecast_mask.any():
        examples = (
            target_preds.loc[
                invalid_forecast_mask
            ]
            .head(20)
        )

        display(examples)

        raise ValueError(
            f"Found {invalid_forecast_mask.sum():,} "
            f"NaN or infinite forecasts for "
            f"{target_date:%Y-%m}."
        )

    expected_skus = set(
        fold["test_df"]["sku_id"]
    )

    predicted_skus = set(
        target_preds["sku_id"]
    )

    known_at_origin = set(
        fold["known_at_origin"]
    )

    post_origin_first_sale_skus = set(
        fold["post_origin_first_sale_skus"]
    )

    missing_skus = (
        expected_skus
        - predicted_skus
    )

    missing_known_skus = (
        missing_skus
        & known_at_origin
    )

    missing_post_origin_skus = (
        missing_skus
        & post_origin_first_sale_skus
    )

    unexplained_missing_skus = (
        missing_skus
        - missing_known_skus
        - missing_post_origin_skus
    )

    if missing_known_skus:
        raise ValueError(
            f"Model failed to forecast "
            f"{len(missing_known_skus):,} SKUs "
            f"that were known at the forecast origin "
            f"{fold['origin']:%Y-%m}."
        )

    if unexplained_missing_skus:
        raise ValueError(
            f"{len(unexplained_missing_skus):,} "
            f"eligible SKUs have unexplained missing "
            f"predictions."
        )

    n_cold_start_skus_defaulted = 0

    if missing_post_origin_skus:
        if cold_start_strategy == "zero":
            cold_start_predictions = pd.DataFrame(
                {
                    "sku_id": sorted(
                        missing_post_origin_skus
                    ),
                    "y_pred": 0.0,
                }
            )

            target_preds = pd.concat(
                [
                    target_preds,
                    cold_start_predictions,
                ],
                ignore_index=True,
            )

            n_cold_start_skus_defaulted = len(
                missing_post_origin_skus
            )

        elif cold_start_strategy == "error":
            raise ValueError(
                f"{len(missing_post_origin_skus):,} "
                f"post-origin first-sale SKUs require "
                f"an explicit cold-start strategy."
            )

        else:
            raise ValueError(
                "cold_start_strategy must be "
                "'zero' or 'error'."
            )

    unexpected_prediction_skus = (
        set(target_preds["sku_id"])
        - expected_skus
    )

    if unexpected_prediction_skus:
        raise ValueError(
            f"The model produced target predictions "
            f"for {len(unexpected_prediction_skus):,} "
            f"SKUs outside the scoring universe."
        )

    merged = (
        fold["test_df"][
            [
                "sku_id",
                "demand",
            ]
        ]
        .merge(
            target_preds,
            on="sku_id",
            how="left",
            validate="one_to_one",
        )
    )

    if merged["y_pred"].isna().any():
        raise AssertionError(
            "Strict forecast join completed with "
            "missing predictions."
        )

    if len(merged) != len(expected_skus):
        raise AssertionError(
            "Strict forecast join changed the number "
            "of eligible SKUs."
        )

    merged["origin"] = fold["origin"]
    merged["target_date"] = target_date

    coverage = {
        "n_expected_skus": len(
            expected_skus
        ),
        "n_post_origin_first_sale_skus": len(
            post_origin_first_sale_skus
        ),
        "n_cold_start_skus_defaulted": (
            n_cold_start_skus_defaulted
        ),
        "n_post_origin_first_sale_skus_not_scored": (
            len(post_origin_first_sale_skus)
            if (
                fold["scoring_policy"]
                == "known_at_origin"
            )
            else 0
        ),
    }

    return merged, coverage
    
# def _score_fold(fold: Dict[str, Any], preds: pd.DataFrame):
#     """Turn one fold's forecast into a row of atomic sums.

#     fold["test_df"] is already restricted to SKUs active at target_date. A SKU
#     active at target but absent from preds launched inside the lag window --
#     the model had no history to forecast it from -- so it's excluded from
#     WMAPE and counted in n_skus_excluded_new, never silently zero-filled.
#     """
#     merged, n_skus_excluded_new = _joined_fold(fold, preds)
#     err = merged["demand"] - merged["y_pred"]
#     abs_actual_sum = float(merged["demand"].abs().sum())

#     return {
#         "origin": fold["origin"],
#         "target_date": fold["target_date"],
#         "n_skus": int(len(merged)),
#         "n_skus_excluded_new": n_skus_excluded_new,
#         "abs_error_sum": float(err.abs().sum()),
#         "abs_actual_sum": abs_actual_sum,
#         "error_sum": float(err.sum()),
#         "wmape": wmape(merged["demand"], merged["y_pred"]),
#     }

def _score_joined_fold(fold, merged, coverage):
    """Calculate atomic WMAPE components from an already validated fold join."""
    error = merged["demand"] - merged["y_pred"]
    abs_actual_sum = float(merged["demand"].abs().sum())
    abs_error_sum = float(error.abs().sum())

    return {
        "origin": fold["origin"],
        "target_date": fold["target_date"],
        "scoring_policy": fold["scoring_policy"],
        "n_skus": len(merged),
        **coverage,
        "abs_error_sum": abs_error_sum,
        "abs_actual_sum": abs_actual_sum,
        "error_sum": float(error.sum()),
        "wmape": abs_error_sum / abs_actual_sum if abs_actual_sum > 0 else np.nan
    }
    
def _score_fold(fold, preds, cold_start_strategy="error"):
    """Validate and score one forecast fold."""
    merged, coverage = _joined_fold(
        fold, preds,
        cold_start_strategy=cold_start_strategy
    )
    return _score_joined_fold(fold, merged, coverage)

# def _score_fold(fold, preds, cold_start_strategy="error"):
#     """
#     Reduce one strictly validated fold to atomic WMAPE components.
#     """
#     merged, coverage = _joined_fold(
#         fold,
#         preds,
#         cold_start_strategy=(
#             cold_start_strategy
#         ),
#     )

#     err = (
#         merged["demand"]
#         - merged["y_pred"]
#     )

#     abs_actual_sum = float(
#         merged["demand"].abs().sum()
#     )

#     return {
#         "origin": fold["origin"],
#         "target_date": fold["target_date"],
#         "scoring_policy": (
#             fold["scoring_policy"]
#         ),
#         "n_skus": int(len(merged)),
#         **coverage,
#         "abs_error_sum": float(
#             err.abs().sum()
#         ),
#         "abs_actual_sum": (
#             abs_actual_sum
#         ),
#         "error_sum": float(
#             err.sum()
#         ),
#         "wmape": wmape(
#             merged["demand"],
#             merged["y_pred"],
#         ),
#     }
    
# def _score_fold_segments(fold, preds, sku_segment, segment_col="sb_class"):
#     """Same join as _score_fold, grouped by segment_col before summing.
#     Tracks n_skus_excluded_new PER SEGMENT, since exclusion rates can differ
#     meaningfully by segment."""
#     target_date = fold["target_date"]
#     target_preds = preds.loc[preds["month"] == target_date, ["sku_id", "y_pred"]].copy()

#     merged = (fold["test_df"][["sku_id", "demand"]]
#               .merge(target_preds, on="sku_id", how="left")
#               .merge(sku_segment[["sku_id", segment_col]], on="sku_id", how="left"))

#     missing_forecast = merged["y_pred"].isna()
#     excluded_counts = merged.loc[missing_forecast].groupby(segment_col).size().rename("n_skus_excluded_new")

#     merged = merged[~missing_forecast].copy()
#     merged["abs_error"] = (merged["demand"] - merged["y_pred"]).abs()

#     out = merged.groupby(segment_col).agg(
#         abs_error_sum=("abs_error", "sum"),
#         abs_actual_sum=("demand", lambda s: s.abs().sum()),
#         n_skus=("sku_id", "count"),
#     ).reset_index()
#     out = out.merge(excluded_counts, on=segment_col, how="left")
#     out["n_skus_excluded_new"] = out["n_skus_excluded_new"].fillna(0).astype(int)
#     out["origin"] = fold["origin"]
#     out["target_date"] = target_date
#     return out

def _score_fold_segments(
    fold,
    preds,
    sku_segment,
    segment_col="sb_class",
    cold_start_strategy="error",
):
    """
    Strict fold scoring split by segment.

    Post-origin first-sale SKUs with no historical segment are assigned
    the explicit segment 'ColdStart' when zero-defaulted.
    """
    merged, coverage = _joined_fold(
        fold,
        preds,
        cold_start_strategy=(
            cold_start_strategy
        ),
    )

    segment_lookup = (
        sku_segment[
            [
                "sku_id",
                segment_col,
            ]
        ]
        .drop_duplicates("sku_id")
    )

    merged = merged.merge(
        segment_lookup,
        on="sku_id",
        how="left",
        validate="many_to_one",
    )

    missing_segment_mask = (
        merged[segment_col].isna()
    )

    if missing_segment_mask.any():
        post_origin_mask = (
            merged["sku_id"].isin(
                fold[
                    "post_origin_first_sale_skus"
                ]
            )
        )

        cold_start_segment_mask = (
            missing_segment_mask
            & post_origin_mask
        )

        merged.loc[
            cold_start_segment_mask,
            segment_col,
        ] = "ColdStart"

        remaining_missing = (
            merged[segment_col].isna()
        )

        if remaining_missing.any():
            raise ValueError(
                f"{remaining_missing.sum():,} "
                f"known SKUs are missing "
                f"{segment_col} labels."
            )

    merged["abs_error"] = (
        merged["demand"]
        - merged["y_pred"]
    ).abs()

    out = (
        merged.groupby(
            segment_col,
            as_index=False,
        )
        .agg(
            abs_error_sum=(
                "abs_error",
                "sum",
            ),
            abs_actual_sum=(
                "demand",
                lambda s: s.abs().sum(),
            ),
            n_skus=(
                "sku_id",
                "count",
            ),
        )
    )

    out["wmape"] = np.where(
        out["abs_actual_sum"] > 0,
        (
            out["abs_error_sum"]
            / out["abs_actual_sum"]
        ),
        np.nan,
    )

    out["origin"] = fold["origin"]
    out["target_date"] = (
        fold["target_date"]
    )
    out["scoring_policy"] = (
        fold["scoring_policy"]
    )

    return out



def run_backtest(
    forecast_fn,
    full_panel,
    active_windows,
    params=None,
    min_train_months=12,
    lag=LAG_MONTHS,
    horizon=HORIZON,
    window_months=None,
    scoring_policy="known_at_origin",
    cold_start_strategy="error",
    on_fold_complete=None,
    collect_predictions=False,
    scope_fn=None,              # NEW -- restricts which SKUs the model sees/is scored on.
                                  # None (default) = no change to current behaviour.
    sku_segment=None,
    segment_col="sb_class",
    verbose=False,
):
    """
    Run a strict rolling-origin backtest.

    Eligibility is determined independently of forecast availability.
    Every eligible SKU must receive a valid target prediction.
    """
    params = params or {}

    rows = []
    prediction_rows = (
        []
        if collect_predictions
        else None
    )

    run_abs_err = 0.0
    run_abs_act = 0.0

    folds = expanding_window_folds(
        full_panel,
        active_windows,
        min_train_months=min_train_months,
        lag=lag,
        horizon=horizon,
        window_months=window_months,
        scoring_policy=scoring_policy,
    )

    for fold_idx, fold in enumerate(folds):
        if scope_fn is not None:
            fold = _apply_scope(fold, scope_fn)          # <-- NEW, only line added here

        preds = forecast_fn(
            fold["train_df"],
            horizon=len(
                fold["horizon_dates"]
            ),
            **params,
        )

        if sku_segment is None:
            row = _score_fold(
                fold,
                preds,
                cold_start_strategy=(
                    cold_start_strategy
                ),
            )

            rows.append(row)

            fold_abs_err = (
                row["abs_error_sum"]
            )
            fold_abs_act = (
                row["abs_actual_sum"]
            )

        else:
            seg_df = _score_fold_segments(
                fold,
                preds,
                sku_segment,
                segment_col=segment_col,
                cold_start_strategy=(
                    cold_start_strategy
                ),
            )

            rows.extend(
                seg_df.to_dict("records")
            )

            fold_abs_err = float(
                seg_df[
                    "abs_error_sum"
                ].sum()
            )
            fold_abs_act = float(
                seg_df[
                    "abs_actual_sum"
                ].sum()
            )

        if collect_predictions:
            merged, _ = _joined_fold(
                fold,
                preds,
                cold_start_strategy=(
                    cold_start_strategy
                ),
            )

            merged["abs_error"] = (
                merged["demand"]
                - merged["y_pred"]
            ).abs()

            if sku_segment is not None:
                merged = merged.merge(
                    sku_segment,
                    on="sku_id",
                    how="left",
                )

            prediction_rows.append(
                merged
            )

        run_abs_err += fold_abs_err
        run_abs_act += fold_abs_act

        if verbose:
            wmape_this_fold = (
                fold_abs_err / fold_abs_act
                if fold_abs_act > 0
                else float("nan")
            )

            print(
                f"fold {fold_idx + 1:>2}  "
                f"origin={fold['origin']:%Y-%m}  "
                f"target={fold['target_date']:%Y-%m}  "
                f"wmape={wmape_this_fold:.4f}"
            )

        if on_fold_complete is not None:
            running_pooled = (
                run_abs_err / run_abs_act
                if run_abs_act > 0
                else float("nan")
            )

            if on_fold_complete(
                fold_idx,
                running_pooled,
            ):
                break

    results_df = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "origin",
                "target_date",
            ]
        )
        .reset_index(drop=True)
    )

    if collect_predictions:
        predictions_df = (
            pd.concat(
                prediction_rows,
                ignore_index=True,
            )
            .sort_values(
                [
                    "sku_id",
                    "origin",
                ]
            )
            .reset_index(drop=True)
        )

        return (
            results_df,
            predictions_df,
        )

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
    if ("n_post_origin_first_sale_skus_not_scored" in results_df.columns):
        n_not_scored = int(results_df["n_post_origin_first_sale_skus_not_scored"].sum())

        if n_not_scored > 0:
            print("Post-origin first-sale SKU-fold "
                f"cases not included in the selected "
                f"scoring universe: {n_not_scored:,}")
            
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


def summarise_segments(full_panel, sku_segment, segment_col="sb_class"):
    """
    Per-segment summary: how many SKUs are in each group, and how much of
    total demand volume they represent. These are NOT the same thing --
    classify_sb only classifies, it doesn't tell you which groups actually
    matter for WMAPE, since WMAPE is volume-weighted, not SKU-count-weighted.
    share_of_demand here is exactly the share_s term from the pooled-WMAPE
    identity (pooled = sum(share_s * wmape_s)) -- this is where those
    weights actually come from.
    """
    total_demand_per_sku = full_panel.groupby("sku_id")["demand"].sum().rename("total_demand").reset_index()
    merged = total_demand_per_sku.merge(sku_segment[["sku_id", segment_col]], on="sku_id", how="left")

    summary = merged.groupby(segment_col).agg(
        n_skus=("sku_id", "count"),
        total_demand=("total_demand", "sum"),
    ).reset_index()

    summary["share_of_skus"] = summary["n_skus"] / summary["n_skus"].sum()
    summary["share_of_demand"] = summary["total_demand"] / summary["total_demand"].sum()

    return summary.sort_values("share_of_demand", ascending=False).reset_index(drop=True)


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


# def illustrate_folds(forecast_fn, full_panel, active_windows, sku_ids, origins,
#                       params=None, lag=LAG_MONTHS, horizon=6, show_computations=True,
#                       ylim=None, save_prefix=None):
#     """
#     Walk through a handful of specific origins, one plot per origin, showing
#     the named SKU(s)' known history, forecast curve, and the exact error at
#     the scored (lag) target month. This is the illustrative "here's how the
#     harness works, and how multiple SKUs pool into one WMAPE" view -- for
#     the full rolling series across every origin, use run_backtest directly.

#     sku_ids : a single sku_id, or a list of 2-4 for a comparison view within
#         each fold (more than ~4 gets visually unreadable).
#     origins : specific origin months to illustrate, e.g.
#         ["2021-06-01", "2021-07-01", "2021-08-01", "2021-09-01"]
#     show_computations : if True (default), adds the right-hand text panel
#         with each SKU's actual/forecast/error and the fold's WMAPE. Set
#         False for a cleaner, presentation-only version of the same plot.
#     """
#     params = params or {}
#     if isinstance(sku_ids, str):
#         sku_ids = [sku_ids]

#     colors = ["#2a78d6", "#e07b39", "#4caf50", "#9c27b0"]
#     origins = sorted(pd.Timestamp(o) for o in origins)

#     if ylim is None:
#         relevant = full_panel[full_panel["sku_id"].isin(sku_ids)]["demand"]
#         ylim = (0, relevant.max() * 1.3)

#     for i, origin in enumerate(origins, start=1):
#         target_date = origin + pd.DateOffset(months=lag)
#         all_months = pd.date_range(full_panel["month"].min(), full_panel["month"].max(), freq="MS")
#         origin_idx = all_months.get_loc(origin)
#         horizon_end_idx = min(origin_idx + horizon, len(all_months) - 1)
#         horizon_dates = all_months[origin_idx + 1: horizon_end_idx + 1]

#         train_df = train_frame(full_panel, origin)
#         scored_skus = active_skus_at(active_windows, target_date)
#         test_df = full_panel.loc[(full_panel["month"] == target_date) & (full_panel["sku_id"].isin(scored_skus))].copy()
        
#         # fold = {"origin": origin, "target_date": target_date, "horizon_dates": horizon_dates,
#         #         "train_df": train_df, "test_df": test_df}

#         # preds = forecast_fn(train_df, horizon=len(horizon_dates), **params)
#         # row = _score_fold(fold, preds)
#         # merged, _ = _joined_fold(fold, preds)
        
#         fold = make_fold(full_panel,active_windows,origin,lag=lag,horizon=horizon,scoring_policy="known_at_origin")

#         preds = forecast_fn(fold["train_df"],horizon=len(fold["horizon_dates"]),**params)
#         row = _score_fold(fold, preds, cold_start_strategy="error")
#         merged, _ = _joined_fold(fold, preds, cold_start_strategy="error")

#         if show_computations:
#             fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [2.3, 1]})
#             ax, ax2 = axes
#         else:
#             fig, ax = plt.subplots(figsize=(11, 4.3))

#         for sku, color in zip(sku_ids, colors):
#             hist = full_panel[(full_panel["sku_id"] == sku) & (full_panel["month"] <= origin)]
#             ax.plot(hist["month"], hist["demand"], marker="o", color=color, label=f"{sku} actual (known)")

#             fc_sku = preds[preds["sku_id"] == sku]
#             ax.plot(fc_sku["month"], fc_sku["y_pred"], marker="s", linestyle="--", color=color,
#                     alpha=0.6, label=f"{sku} forecast")

#             act_row = full_panel[(full_panel["sku_id"] == sku) & (full_panel["month"] == target_date)]
#             fc_row = fc_sku[fc_sku["month"] == target_date]
#             if len(act_row) and len(fc_row):
#                 a, f = act_row["demand"].values[0], fc_row["y_pred"].values[0]
#                 ax.plot([target_date, target_date], [a, f], color=color, linewidth=3, zorder=5)
#                 ax.scatter([target_date], [a], color=color, s=70, zorder=6, edgecolor="black")
#                 ax.scatter([target_date], [f], color=color, marker="s", s=70, zorder=6, edgecolor="black")
#                 ax.annotate(f"|err|={abs(a-f):.2f}", (target_date, max(a, f) + ylim[1]*0.03),
#                             ha="center", fontsize=8, color=color)

#         ax.axvline(origin, color="black", linestyle=":", linewidth=1.2)
#         ax.axvline(target_date, color="gray", linestyle=":", linewidth=1.2)
#         ax.set_ylim(*ylim)
#         ax.set_title(f"Fold {i}: origin={origin.strftime('%b %Y')}  ->  scored at {target_date.strftime('%b %Y')}")
#         ax.legend(fontsize=7, loc="lower left")

#         if show_computations:
#             ax2.axis("off")
#             lines = [f"Target: {target_date.strftime('%b %Y')}", ""]
#             for r in merged[merged["sku_id"].isin(sku_ids)].itertuples():
#                 lines.append(f"{r.sku_id}: actual={r.demand:.0f}  forecast={r.y_pred:.2f}  "
#                              f"|err|={abs(r.demand - r.y_pred):.2f}")
#             lines += ["", f"sum |errors| (all scored SKUs) = {row['abs_error_sum']:.3f}",
#                       f"sum |actuals| (all scored SKUs) = {row['abs_actual_sum']:.3f}", "",
#                       f"WMAPE = {row['wmape']:.4f}"]
#             ax2.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=10.5, family="monospace")

#         plt.tight_layout()
#         if save_prefix:
#             plt.savefig(f"{save_prefix}_fold{i}_{origin.strftime('%b').lower()}.png", bbox_inches="tight")
#         plt.show()
#         print(f"  Fold {i}: origin={origin.date()}  target={target_date.date()}  WMAPE={row['wmape']:.4f}")

def illustrate_folds(
    forecast_fn, full_panel, active_windows, sku_ids, origins, params=None,
    lag=LAG_MONTHS, horizon=6, window_months=None,
    scoring_policy="known_at_origin", cold_start_strategy="error",
    show_computations=True, ylim=None, save_prefix=None
):
    """
    Illustrate selected rolling-origin folds.

    Each plot shows:
    - demand history available at the forecast origin;
    - the model's forecast horizon;
    - actual and forecast demand at the scored target month;
    - the target-month absolute error for each requested SKU;
    - optionally, the pooled WMAPE across the complete scoring universe.

    Parameters
    ----------
    forecast_fn : callable
        Forecast function with signature forecast_fn(train_df, horizon, **params).

    full_panel : pd.DataFrame
        Monthly SKU demand panel.

    active_windows : pd.DataFrame
        SKU first-observed and last-observed dates.

    sku_ids : str or list[str]
        SKUs to display.

    origins : iterable
        Forecast-origin months to illustrate.

    scoring_policy : {"known_at_origin", "all_observed_by_target"}
        Evaluation-universe policy.

    cold_start_strategy : {"error", "zero"}
        Treatment of eligible post-origin first-sale SKUs without forecasts.

    Notes
    -----
    The fold is constructed by make_fold(), ensuring that this visualisation
    uses the same eligibility and scoring rules as run_backtest().
    """
    params = params or {}
    sku_ids = [sku_ids] if isinstance(sku_ids, str) else list(sku_ids)
    origins = sorted(pd.Timestamp(origin).to_period("M").to_timestamp() for origin in origins)

    if not sku_ids:
        raise ValueError("Provide at least one SKU.")

    if horizon < lag:
        raise ValueError(
            f"horizon={horizon} does not reach the scored target at lag={lag}. "
            f"Use horizon >= {lag}."
        )

    missing_skus = set(sku_ids) - set(full_panel["sku_id"])
    if missing_skus:
        raise ValueError(f"Requested SKUs are not present in full_panel: {sorted(missing_skus)}")

    colors = ["#2a78d6", "#e07b39", "#4caf50", "#9c27b0", "#795548", "#00acc1"]

    if len(sku_ids) > len(colors):
        raise ValueError(f"illustrate_folds supports up to {len(colors)} SKUs per plot.")

    if ylim is None:
        relevant = full_panel.loc[full_panel["sku_id"].isin(sku_ids), "demand"]
        upper = max(float(relevant.max()) * 1.3, 1.0)
        ylim = (0, upper)

    for fold_number, origin in enumerate(origins, start=1):
        fold = make_fold(
            full_panel, active_windows, origin,
            lag=lag, horizon=horizon, window_months=window_months,
            scoring_policy=scoring_policy
        )

        preds = forecast_fn(
            fold["train_df"],
            horizon=len(fold["horizon_dates"]),
            **params
        )

        merged, coverage = _joined_fold(
            fold, preds,
            cold_start_strategy=cold_start_strategy
        )

        row = _score_joined_fold(fold, merged, coverage)

        if show_computations:
            fig, (ax, ax2) = plt.subplots(
                1, 2, figsize=(13, 4.3),
                gridspec_kw={"width_ratios": [2.3, 1]}
            )
        else:
            fig, ax = plt.subplots(figsize=(11, 4.3))
            ax2 = None

        known_skus = set(fold["known_at_origin"])
        scored_skus = set(fold["test_df"]["sku_id"])

        for sku, color in zip(sku_ids, colors):
            sku_known = sku in known_skus
            sku_scored = sku in scored_skus

            history = full_panel.loc[
                (full_panel["sku_id"] == sku) &
                (full_panel["month"] <= fold["origin"])
            ].sort_values("month")

            if not history.empty:
                ax.plot(
                    history["month"], history["demand"],
                    marker="o", color=color,
                    label=f"{sku} actual history"
                )

            sku_preds = preds.loc[preds["sku_id"] == sku].sort_values("month")

            if not sku_preds.empty:
                ax.plot(
                    sku_preds["month"], sku_preds["y_pred"],
                    marker="s", linestyle="--", color=color, alpha=0.65,
                    label=f"{sku} forecast"
                )

            scored_row = merged.loc[merged["sku_id"] == sku]

            if not scored_row.empty:
                actual = float(scored_row["demand"].iloc[0])
                forecast = float(scored_row["y_pred"].iloc[0])
                error = abs(actual - forecast)
                target = fold["target_date"]

                ax.plot(
                    [target, target], [actual, forecast],
                    color=color, linewidth=3, zorder=5
                )
                ax.scatter(
                    target, actual, color=color, s=70,
                    edgecolor="black", zorder=6
                )
                ax.scatter(
                    target, forecast, color=color, marker="s", s=70,
                    edgecolor="black", zorder=6
                )
                ax.annotate(
                    f"|err|={error:.2f}",
                    (target, max(actual, forecast) + ylim[1] * 0.03),
                    ha="center", fontsize=8, color=color
                )

            elif not sku_known:
                print(
                    f"Fold {fold_number}, origin {origin:%Y-%m}: "
                    f"{sku} had no observed history at the origin, "
                    "so no standard forecast was available."
                )
            elif not sku_scored:
                print(
                    f"Fold {fold_number}, origin {origin:%Y-%m}: "
                    f"{sku} was not included under scoring_policy="
                    f"'{scoring_policy}'."
                )

        ax.axvline(
            fold["origin"], color="black",
            linestyle=":", linewidth=1.2, label="forecast origin"
        )
        ax.axvline(
            fold["target_date"], color="gray",
            linestyle=":", linewidth=1.2, label="scored target"
        )

        ax.set_ylim(*ylim)
        ax.set_title(
            f"Fold {fold_number}: origin={fold['origin']:%b %Y} "
            f"→ scored at {fold['target_date']:%b %Y}"
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Demand")
        ax.legend(fontsize=7, loc="lower left")

        if show_computations:
            ax2.axis("off")
            requested_rows = merged.loc[merged["sku_id"].isin(sku_ids)]
            lines = [
                f"Origin: {fold['origin']:%b %Y}",
                f"Target: {fold['target_date']:%b %Y}",
                f"Policy: {scoring_policy}",
                ""
            ]

            for result in requested_rows.itertuples():
                error = abs(result.demand - result.y_pred)
                lines.append(
                    f"{result.sku_id}:\n"
                    f"  actual={result.demand:.0f}\n"
                    f"  forecast={result.y_pred:.2f}\n"
                    f"  |error|={error:.2f}"
                )

            requested_not_scored = [sku for sku in sku_ids if sku not in set(requested_rows["sku_id"])]
            for sku in requested_not_scored:
                status = (
                    "not observed at origin"
                    if sku not in known_skus
                    else "not in scoring universe"
                )
                lines.append(f"{sku}:\n  {status}")

            lines.extend([
                "",
                "Complete scoring universe:",
                f"  SKUs={row['n_skus']:,}",
                f"  sum |errors|={row['abs_error_sum']:.3f}",
                f"  sum |actuals|={row['abs_actual_sum']:.3f}",
                f"  WMAPE={row['wmape']:.4f}"
            ])

            if coverage["n_cold_start_skus_defaulted"]:
                lines.append(
                    f"  cold-start zeros="
                    f"{coverage['n_cold_start_skus_defaulted']:,}"
                )

            ax2.text(
                0.02, 0.97, "\n".join(lines),
                va="top", fontsize=9.5, family="monospace"
            )

        plt.tight_layout()

        if save_prefix:
            filename = f"{save_prefix}_fold{fold_number}_{origin:%Y-%m}.png"
            plt.savefig(filename, bbox_inches="tight")

        plt.show()

        print(
            f"Fold {fold_number}: origin={fold['origin'].date()}  "
            f"target={fold['target_date'].date()}  "
            f"n_skus={row['n_skus']:,}  WMAPE={row['wmape']:.4f}"
        )


# Same palette already used across the project's own plotting functions
COLOR_MONTHLY = "#ccc"      # noisy monthly signal -- background context, same role as "full history"
COLOR_ROLLING = "#2a78d6"   # the smoothed number that matters -- same blue used for "actual"

def plot_wmape_over_time(
    results_df,
    window: int = 3,
    title: str = "Naive baseline: pooled WMAPE through time",
):
    """
    results_df: the *pooled* output of run_backtest — i.e. called WITHOUT
    sku_segment, one row per fold. summarise_backtest() expects the same
    shape, so anything you've already run through that function can go
    straight into this one.

    If you ran a segmented backtest instead (sku_segment=sb_lookup, one row
    per fold+segment), collapse it back to one row per fold first:

        pooled = (
            segmented_results
            .groupby(["origin", "target_date"], as_index=False)
            [["abs_error_sum", "abs_actual_sum"]].sum()
        )
        pooled["wmape"] = pooled["abs_error_sum"] / pooled["abs_actual_sum"]

    Segments are disjoint, so summing atomic sums first reconstructs the
    pooled total exactly (same identity the leverage-argument slide uses).
    Don't feed segmented rows straight into this function —
    rolling_average_wmape assumes one row per fold, so a segmented frame
    would roll over segment-rows instead of folds.
    """
    r = results_df.sort_values("origin").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(
        r["target_date"],
        r["wmape"],
        color=COLOR_MONTHLY,
        linewidth=1.3,
        marker="o",
        markersize=3,
        label="Monthly pooled WMAPE",
    )
    ax.plot(
        r["target_date"],
        rolling_average_wmape(r, window=window, method="pooled"),
        color=COLOR_ROLLING,
        linewidth=2.2,
        label=f"{window}-month rolling pooled WMAPE",
    )

    ax.set_title(title)
    ax.set_xlabel("Target month (scored)")
    ax.set_ylabel("WMAPE")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig



# Same 4-colour palette illustrate_folds() uses for multi-series comparison,
# assigned in descending share-of-demand order (Smooth largest -> Lumpy smallest)
SEGMENT_COLORS = {
    "Smooth": "#2a78d6",
    "Erratic": "#e07b39",
    "Intermittent": "#4caf50",
    "Lumpy": "#9c27b0",
}


def plot_leverage_argument(
    segmented_results,
    segment_summary,
    segment_col: str = "sb_class",
    head_segments=("Smooth", "Erratic"),
    tail_segments=("Intermittent", "Lumpy"),
    cut: float = 0.10,
    title: str = "The leverage argument: same effort, different payoff",
):
    """
    segmented_results: output of
        run_backtest(forecast_fn, full_panel, windows,
                      sku_segment=sb_lookup, segment_col=segment_col)
    -- one row per (fold, segment).

    segment_summary: the DataFrame returned by
        summarise_segments(full_panel, sb_lookup, segment_col=segment_col)
    -- must contain [segment_col, "share_of_demand"].

    Draws three stacked bars, each stacked by segment. Each segment's
    slice height is share_of_demand * that segment's WMAPE -- literally one
    term of the pooled-WMAPE identity -- so the total bar height is the
    pooled WMAPE for that scenario, and the stack composition shows where
    it's coming from:

      1. Current  -- pooled WMAPE as actually backtested.
      2. A `cut` (default 10 points) applied ONLY to head_segments.
      3. The SAME `cut` applied ONLY to tail_segments.

    Bars 2 and 3 use an identical cut size, so the height difference
    between them is not an illustration of the leverage argument, it IS
    the leverage argument -- same modelling effort, different payoff,
    purely from where volume sits.
    """
    current = pooled_wmape(segmented_results, by=segment_col)
    merged = current.merge(
        segment_summary[[segment_col, "share_of_demand"]], on=segment_col
    )
    merged = merged[merged["share_of_demand"] > 0].reset_index(drop=True)

    def contributions(wmape_cuts=None):
        wmape = merged.set_index(segment_col)["pooled_wmape"].copy()
        if wmape_cuts:
            for seg, delta in wmape_cuts.items():
                if seg in wmape.index:
                    wmape[seg] = max(wmape[seg] - delta, 0.0)
        share = merged.set_index(segment_col)["share_of_demand"]
        return (share * wmape).rename("contribution")

    scenario_labels = [
        "Current\n(as backtested)",
        f"-{int(cut*100)}pt cut:\nSmooth + Erratic",
        f"-{int(cut*100)}pt cut:\nIntermittent + Lumpy",
    ]

    scenarios = {
        scenario_labels[0]: contributions(),
        scenario_labels[1]: contributions({s: cut for s in head_segments}),
        scenario_labels[2]: contributions({s: cut for s in tail_segments}),
    }

    segment_order = merged.sort_values("share_of_demand", ascending=False)[segment_col].tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(scenarios))
    bottoms = [0.0] * len(scenarios)

    for seg in segment_order:
        heights = [scenarios[label][seg] for label in scenarios]
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=seg,
            color=SEGMENT_COLORS.get(seg, "#999999"),
            width=0.55,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    for i, label in enumerate(scenarios):
        total = float(scenarios[label].sum())
        ax.text(i, total + 0.012, f"{total:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(list(scenarios.keys()))
    ax.set_ylabel("Pooled WMAPE\n(stacked: segment share of demand x segment WMAPE)")
    ax.set_title(title)
    ax.legend(title="SB class", fontsize=8)
    plt.tight_layout()

    return fig




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


def scope_to(*allowed_classes, classify_fn=classify_sb, segment_col="sb_class"):
    """
    Builds a scope_fn for run_backtest(scope_fn=...). Classification happens
    inside, on whatever train_df it's given each fold -- so it inherits
    point-in-time correctness for free, it never needs to know what fold
    it's in.
    """
    allowed = set(allowed_classes)

    def scope_fn(train_df):
        seg = classify_fn(train_df)
        return set(seg.loc[seg[segment_col].isin(allowed), "sku_id"])

    return scope_fn

def assert_scopes_exhaustive(full_panel, active_windows, origin, scope_fns,
                              classify_fn=classify_sb, segment_col="sb_class",
                              window_months=None):
    """
    Confirms a set of scope_fns jointly cover every SKU known at origin, and
    don't overlap. Run this once per pair of branches you build, not on
    every backtest call.
    """
    train_df = train_frame(full_panel, origin, window_months=window_months)
    known = active_skus_at(active_windows, origin)

    scoped_sets = [sf(train_df) for sf in scope_fns]
    union = set().union(*scoped_sets)
    overlap = set()
    for i in range(len(scoped_sets)):
        for j in range(i + 1, len(scoped_sets)):
            overlap |= scoped_sets[i] & scoped_sets[j]

    missing = known - union
    assert not missing, f"{len(missing)} known-at-origin SKUs fall in no scope: {list(missing)[:10]}"
    assert not overlap, f"{len(overlap)} SKUs fall in more than one scope: {list(overlap)[:10]}"
    return True


def segment_scope(forecast_fn, allowed_classes, classify_fn=classify_sb, segment_col="sb_class"):
    """
    Wraps a plain forecast_fn(train_df, horizon, **kwargs) -> DataFrame so it
    only ever sees and predicts for SKUs in allowed_classes, classified from
    train_df itself -- so classification is automatically point-in-time,
    since train_df is already cut off at the fold's origin by train_frame().

    forecast_fn itself stays completely segment-agnostic; it never needs to
    know classes exist.
    """
    allowed = set(allowed_classes)

    def wrapped(train_df, horizon, **kwargs):
        segment_lookup = classify_fn(train_df)  # e.g. DataFrame[sku_id, sb_class]
        scoped_skus = segment_lookup.loc[
            segment_lookup[segment_col].isin(allowed), "sku_id"
        ]
        scoped_train_df = train_df[train_df["sku_id"].isin(scoped_skus)]

        if scoped_train_df.empty:
            return pd.DataFrame(columns=["sku_id", "month", "y_pred"])

        return forecast_fn(scoped_train_df, horizon, **kwargs)

    wrapped.__name__ = f"{forecast_fn.__name__}__scoped_to_{'_'.join(sorted(allowed))}"
    return wrapped


# def combine_segment_results(*results_dfs, on=("origin", "target_date")):
#     """
#     Combines disjoint, exhaustive run_backtest outputs (e.g. head-model
#     results + tail-model results) into one pooled results_df per fold.
#     Sums abs_error_sum/abs_actual_sum BEFORE dividing -- never averages the
#     separate wmape columns, since that would silently equal-weight the two
#     models regardless of how much volume each one actually scored.
#     """
#     combined = results_dfs[0][[*on, "abs_error_sum", "abs_actual_sum"]].copy()

#     for r in results_dfs[1:]:
#         combined = combined.merge(
#             r[[*on, "abs_error_sum", "abs_actual_sum"]],
#             on=list(on), suffixes=("", "_other"), how="outer",
#         )
#         combined["abs_error_sum"] = combined["abs_error_sum"].fillna(0) + combined["abs_error_sum_other"].fillna(0)
#         combined["abs_actual_sum"] = combined["abs_actual_sum"].fillna(0) + combined["abs_actual_sum_other"].fillna(0)
#         combined = combined.drop(columns=["abs_error_sum_other", "abs_actual_sum_other"])

#     combined["wmape"] = combined["abs_error_sum"] / combined["abs_actual_sum"]
#     return combined.sort_values(list(on)).reset_index(drop=True)

def combine_segment_results(*results_dfs, on=("origin", "target_date")):
    """
    Combines disjoint, exhaustive run_backtest outputs (e.g. head-model
    results + tail-model results) into one pooled results_df per fold.

    Each input is first collapsed to exactly one row per fold by summing
    atomic sums across any segment rows. No-op if the input already has
    one row per fold (run_backtest called WITHOUT sku_segment=). Required
    if it was called WITH sku_segment= (one row per fold+class instead).
    """
    def _collapse_to_one_row_per_fold(df):
        return df.groupby(list(on), as_index=False)[["abs_error_sum", "abs_actual_sum"]].sum()

    combined = _collapse_to_one_row_per_fold(results_dfs[0])
    for r in results_dfs[1:]:
        r_collapsed = _collapse_to_one_row_per_fold(r)
        combined = combined.merge(r_collapsed, on=list(on), suffixes=("", "_other"), how="outer")
        combined["abs_error_sum"] = combined["abs_error_sum"].fillna(0) + combined["abs_error_sum_other"].fillna(0)
        combined["abs_actual_sum"] = combined["abs_actual_sum"].fillna(0) + combined["abs_actual_sum_other"].fillna(0)
        combined = combined.drop(columns=["abs_error_sum_other", "abs_actual_sum_other"])

    combined["wmape"] = combined["abs_error_sum"] / combined["abs_actual_sum"]
    return combined.sort_values(list(on)).reset_index(drop=True)


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
