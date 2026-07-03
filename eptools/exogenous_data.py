import os
import pandas as pd
from eptools.data_preprocessing import _DATAFRAMES_CACHE, _resolve_data_path, _freeze


def get_suzuki_sales_post_2014(data_path=None) -> pd.DataFrame:
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


_SOURCES = {
    "suzuki_sales_monthly_post_2014": get_suzuki_sales_post_2014,
}


def get_exog(name=None):
    """
    Discover or fetch exogenous data sources.

    Called with no arguments, returns a list of available source names.
    Called with name=<source>, returns the corresponding DataFrame.

    Args:
        name: Name of the data source to fetch. If None, lists available sources.

    Returns:
        List of source name strings (when name is None), or a DataFrame.

    Example — list sources:
        get_exog()                     # ['suzuki_sales']

    Example — fetch one source:
        df = get_exog(name='suzuki_sales')

    Example — fetch all sources:
        for name in get_exog():
            df = get_exog(name=name)
    """
    if name is None:
        return list(_SOURCES)
    if name not in _SOURCES:
        raise ValueError(f"Unknown source {name!r}. Available: {list(_SOURCES)}")
    return _SOURCES[name]()
