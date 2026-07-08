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
    sales["demand"] = pd.to_numeric(sales["value"], errors="coerce").fillna(0)

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

    sales = sales[sales["is_collision"]]

    return sales.drop(columns=["collision_flag_clean", "is_collision"])

# get the dictionary and strip out the redundant column information
def get_collision_sales_dictionary():
    dfs = load_dataframes()
    dictionary = dfs['dictionary']
    return dictionary[~dictionary["column_name"].isin(REDUNDANT_COLUMNS)]


        
    



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
    


def get_bare_sku_df(sku_code, format=None, no_warnings=False, _sales=None):
    """ return a dataframe with a dateTime index for our period of enquiry and the demand for our period of enquiry
        zero values will be filled for any missing time periods

        _sales: optionally pass a pre-loaded sales DataFrame to avoid repeated copying when
                calling this function in a loop (load_dataframes() copies the full DataFrame
                on every call, so passing it in pays that cost once instead of per-SKU)
    """

    all_months = get_all_months()

    sales = _sales if _sales is not None else load_dataframes()['sales']

    sku_sales = (
        sales[['Date', 'value', 'ts_id']]
        .query('ts_id == @sku_code')
        .assign(Date=lambda df: pd.to_datetime(df['Date']))
        .set_index('Date')[['value']]
        .reindex(all_months, fill_value=0)
        .rename(columns={'value': 'y'})
    )

    if len(sku_sales) == 0:
        raise ValueError(f'SKU not found: {sku_code}')

    if not no_warnings:
        missing_months = (sku_sales['y'] == 0).sum()
        if missing_months > 0:
            print(f"WARNING: {missing_months} months were missing data and defaulted to 0")

    return sku_sales


def get_bodywork_skus():
    all_skus_flagged_as_bodywork = load_dataframes()["sales"].query('FAMILY_DESCRIPTION == "CARROCERIA"')['ts_id'].unique()
    return all_skus_flagged_as_bodywork


def get_modelling_skus(_sales=None):
    """Single source of truth for the modelling SKU universe.

    Edwin's scope (collision_demand_forecasting_edwin_v3, section 4): every SKU that is
    *ever* labelled COLLISION, kept with its full history. The collision flag describes
    what kind of part a SKU is, not what kind of transaction a row is, so a SKU only
    labelled COLLISION from the Jan-2024 enrichment onwards still keeps its pre-2024
    history. This replaces the old FAMILY_DESCRIPTION == "CARROCERIA" proxy, which happens
    to select the same 8,097 SKUs today but is an incidental match, not the definition.

    NOTE: the final modelling scope is still pending client confirmation. Change it here,
    in this one place, rather than throughout the harness.
    """
    sales = _sales if _sales is not None else load_dataframes()["sales"]
    flag = sales["collision_flag"].astype(str).str.strip().str.upper()
    is_collision = flag.str.contains("COLLISION") & ~flag.str.contains("NON")
    ever = is_collision.groupby(sales["ts_id"].astype(str)).any()
    return ever[ever].index.tolist()



def rebuild_nixtla_df():
    """Build the nixtla-format frame (unique_id, ds, y) for the modelling SKU universe
    and cache it to the DATA directory.

    Scope: get_modelling_skus() — ever-COLLISION SKUs (Edwin's section 4).

    Densification is *within each SKU's active window* only — from its first observed
    month to its last — never across the global calendar. This is deliberate: it keeps
    each SKU's real launch/discontinuation boundaries intact, so get_sku_active_windows
    downstream recovers true first_seen/last_seen and the backtest's new-SKU exclusion
    policy has something real to act on. A global zero-fill would make every SKU look
    active from the dataset start and silently defeat that exclusion.

    On the current data the within-window fill count is zero (the panel is already dense
    inside each window), but the reindex + assertion stay as a guard for future refreshes.
    """
    sales = load_dataframes()["sales"]
    skus = get_modelling_skus(_sales=sales)

    sub = sales.loc[sales["ts_id"].astype(str).isin(skus), ["ts_id", "Date", "value"]].copy()
    sub["unique_id"] = sub["ts_id"].astype(str)
    sub["ds"] = pd.to_datetime(sub["Date"]).dt.to_period("M").dt.to_timestamp()
    sub["y"] = pd.to_numeric(sub["value"], errors="coerce").fillna(0.0)

    # collapse any accidental duplicate (unique_id, ds) rows defensively
    observed = sub.groupby(["unique_id", "ds"], as_index=False)["y"].sum()

    # within-window dense index: first..last observed month per SKU
    bounds = observed.groupby("unique_id")["ds"].agg(first="min", last="max").reset_index()
    full_index = pd.concat(
        [pd.DataFrame({"unique_id": r.unique_id,
                       "ds": pd.date_range(r.first, r.last, freq="MS")})
         for r in bounds.itertuples()],
        ignore_index=True,
    )
    nixtla = full_index.merge(observed, on=["unique_id", "ds"], how="left")
    n_filled = int(nixtla["y"].isna().sum())
    nixtla["y"] = nixtla["y"].fillna(0.0)

    expected = int(((bounds["last"].dt.year - bounds["first"].dt.year) * 12
                    + (bounds["last"].dt.month - bounds["first"].dt.month) + 1).sum())
    assert len(nixtla) == expected, (
        f"rebuild_nixtla_df row mismatch: expected {expected:,}, got {len(nixtla):,}"
    )

    nixtla = (nixtla[["unique_id", "ds", "y"]]
              .sort_values(["unique_id", "ds"]).reset_index(drop=True))

    out_path = os.path.join(_resolve_data_path(), "API_SOURCES", "API_SOURCE_nixtla.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nixtla.to_parquet(out_path, index=False)

    print(f"rebuild_nixtla_df: {nixtla['unique_id'].nunique():,} SKUs, {len(nixtla):,} rows, "
          f"filled {n_filled:,} within-window gaps.")
    return nixtla


def get_nixtla_df():
    """Load the pre-built nixtla DataFrame from disk.

    Run rebuild_nixtla_df() first to generate the file.
    """
    path = os.path.join(_resolve_data_path(), 'API_SOURCES', 'API_SOURCE_nixtla.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nixtla parquet not found at {path}. Run rebuild_nixtla_df() to generate it."
        )
    return pd.read_parquet(path)



get_nixtla_df().columns
