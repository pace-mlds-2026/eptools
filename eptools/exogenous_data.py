import os
import pandas as pd
from eptools.data_preprocessing import _DATAFRAMES_CACHE, _resolve_data_path, _freeze


def get_vehicle_sales(data_path=None) -> pd.DataFrame:
    """
    Load the ANAC Suzuki light/medium monthly vehicle sales data.

    Results are cached in memory after the first call. Subsequent calls with
    the same path return a fresh editable copy without re-reading from disk.

    Auto-detects the environment (Colab, Mac, Windows). If auto-detection
    fails, set the EPTOOLS_DATA_PATH environment variable or pass data_path
    explicitly.

    Args:
        data_path: Optional path to the DATA directory. Overrides auto-detection.

    Returns:
        DataFrame loaded from ANAC-vehicle-sales/suzuki_light_medium_monthly.csv
        with a DatetimeIndex on the Date column.
    """
    resolved = _resolve_data_path(data_path)
    cache_key = resolved + "/__anac_vehicle_sales__"

    if cache_key in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE[cache_key].copy()

    file_path = os.path.join(resolved, "ANAC-vehicle-sales", "suzuki_light_medium_monthly.csv")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    _DATAFRAMES_CACHE[cache_key] = _freeze(df)
    return _DATAFRAMES_CACHE[cache_key].copy()
