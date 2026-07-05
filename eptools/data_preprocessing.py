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


def rebuild_nixtla_df():
    """ create a nixtla format dataframe for all relevant skus and save into the data directory
    # this is the nixtla format needed for TimeGPT, TSB etc
    # Column		NameData				TypeDescription
    # unique_id		String or NumberA 		distinct identifier that distinguishes one individual time series from another (e.g., store ID, product code, or stock ticker)
    # ds			Datestamp or Integer	The time index. Usually represented as dates in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.
    # y				Numeric					The actual historical or target value you want to forecast or analyze.
    """

    bodywork_skus = get_bodywork_skus()

    # load once — avoids copying the full sales DataFrame once per SKU
    sales = load_dataframes()['sales']

    frames = []
    for bodywork_sku in bodywork_skus:
        temp_df = get_bare_sku_df(bodywork_sku, no_warnings=True, _sales=sales)
        temp_df['unique_id'] = bodywork_sku
        frames.append(temp_df)

    # ds, y, unique_id must all be columns (not index) for the nixtla format
    nixtla = pd.concat(frames, sort=False)
    nixtla.index = nixtla.index.rename('ds')
    nixtla = nixtla.reset_index()

    out_path = os.path.join(_resolve_data_path(), 'API_SOURCES', 'API_SOURCE_nixtla.parquet')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nixtla.to_parquet(out_path, index=False)

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
