import os
import pandas as pd


def load_dataframes() -> dict:
    """
    Load the raw project data files.

    Works in Google Colab (mounts Shared Drive) and locally when the
    Shared Drive is synced via the Google Drive desktop app.

    Returns a dict with keys:
        'sales'      -> chile_suzuki_historical_sales.csv as a DataFrame
        'dictionary' -> forecasting_data_dictionary.xlsx as a DataFrame
    """
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        data_path = "/content/drive/Shared drives/EmployerProject/DATA"
    except ImportError:
        # __file__ works in an installed module; fall back to cwd in a notebook
        try:
            _base = os.path.dirname(__file__)
        except NameError:
            _base = os.getcwd()
        data_path = os.path.join(_base, "..", "..", "DATA")

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
