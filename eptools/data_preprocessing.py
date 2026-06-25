import glob
import os
import platform
import string
import pandas as pd


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


def load_dataframes(data_path=None) -> dict:
    """
    Load the raw project data files.

    Auto-detects the environment (Colab, Mac, Windows). If auto-detection
    fails, set the EPTOOLS_DATA_PATH environment variable or pass data_path
    explicitly.

    Args:
        data_path: Optional path to the DATA directory. Overrides auto-detection.

    Returns a dict with keys:
        'sales'      -> chile_suzuki_historical_sales.csv as a DataFrame
        'dictionary' -> forecasting_data_dictionary.xlsx as a DataFrame
    """
    if data_path is None:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            data_path = "/content/drive/Shared drives/EmployerProject/DATA"
        except ImportError:
            data_path = _find_local_data_path()
            if data_path is None:
                raise EnvironmentError(
                    "Could not find the DATA directory automatically. "
                    "Set the EPTOOLS_DATA_PATH environment variable, "
                    "or pass data_path explicitly: "
                    "load_dataframes(data_path='/path/to/DATA')"
                )

    sales_df      = pd.read_csv(os.path.join(data_path, "chile_suzuki_historical_sales.csv"))
    dictionary_df = pd.read_excel(os.path.join(data_path, "forecasting_data_dictionary.xlsx"))

    return {"sales": sales_df, "dictionary": dictionary_df}


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


def clean_df(df):
    """
    Apply standard cleaning to the raw sales DataFrame:
      - Drop redundant columns (constant or entirely null)
      - Parse the Date column to datetime

    Args:
        df: Raw sales DataFrame from load_dataframes()['sales']

    Returns:
        Cleaned DataFrame (copy — original is not modified)
    """
    result = df.copy()
    result = result.drop(columns=[c for c in REDUNDANT_COLUMNS if c in result.columns])
    result["Date"] = pd.to_datetime(result["Date"])
    return result
