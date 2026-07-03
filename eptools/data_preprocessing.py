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

