import os
import pandas as pd
from eptools.data_preprocessing import _DATAFRAMES_CACHE, _resolve_data_path, _freeze


def _load_source(relative_path, cache_key, date_format=None, data_path=None) -> pd.DataFrame:
    """
    Generic cached CSV loader for exogenous data sources.

    Resolves the data path, checks the in-memory cache, reads the CSV,
    parses the 'date' column as a DatetimeIndex, freezes the result, and
    returns a mutable copy.

    Args:
        relative_path: Path to the CSV relative to the resolved DATA directory.
        cache_key:     Unique string used as the in-memory cache key.
        date_format:   Optional strptime format string passed to pd.to_datetime.
                       Omit for standard ISO / mixed date strings.
        data_path:     Optional override for the DATA directory.

    Returns:
        DataFrame with a DatetimeIndex on the 'date' column.
    """
    resolved = _resolve_data_path(data_path)
    key = resolved + cache_key
    if key in _DATAFRAMES_CACHE:
        return _DATAFRAMES_CACHE[key].copy()
    df = pd.read_csv(os.path.join(resolved, relative_path))
    df['date'] = pd.to_datetime(df['date'].astype(str), format=date_format)
    df = df.set_index('date')
    _DATAFRAMES_CACHE[key] = _freeze(df)
    return _DATAFRAMES_CACHE[key].copy()


def get_suzuki_vehicle_sales_monthly_post_2014(data_path=None) -> pd.DataFrame:
    """Load ANAC Suzuki monthly vehicle sales data (2014+)."""
    return _load_source(
        "API_SOURCES/API_SOURCE_vehicle_sales_monthly_2014_onwards.csv",
        "__anac_vehicle_sales__",
        data_path=data_path,
    )


def get_suzuki_vehicle_sales_annual_2008_2013(data_path=None) -> pd.DataFrame:
    """Load ANAC Suzuki annual vehicle sales data (2008–2013)."""
    return _load_source(
        "API_SOURCES/API_SOURCE_vehicle_sales_annual_2008_2013.csv",
        "__anac_vehicle_sales_annual_2008_2013__",
        date_format="%Y",
        data_path=data_path,
    )


def get_weather(data_path=None) -> pd.DataFrame:
    """Load central Santiago weather data."""
    return _load_source(
        "API_SOURCES/API_SOURCE_central_weather.csv",
        "__weather__",
        data_path=data_path,
    )


def get_daylight(data_path=None) -> pd.DataFrame:
    """Load pre-expanded Santiago daylight data (2000–2030)."""
    return _load_source(
        "API_SOURCES/API_SOURCE_santiago_daylight.csv",
        "__daylight__",
        data_path=data_path,
    )


def get_parc_allbrands(data_path=None) -> pd.DataFrame:
    """Load INE all brand parc data (2018–2024)."""
    return _load_source(
        "API_SOURCES/API_SOURCE_chile_parc.csv",
        "__anac_vehicle_sales_annual_2018_2024__",
        date_format="%Y",
        data_path=data_path,
    )


def get_macros(data_path=None) -> pd.DataFrame:
    """Load macroeconomic indicator data for Chile."""
    return _load_source(
        "API_SOURCES/API_SOURCE_macros-gaps-filled.csv",
        "__macros__",
        data_path=data_path,
    )


_SOURCES = {
    "macros": get_macros,
    "suzuki_vehicle_sales_monthly_post_2014": get_suzuki_vehicle_sales_monthly_post_2014,
    "suzuki_vehicle_sales_annual_2008_2013": get_suzuki_vehicle_sales_annual_2008_2013,
    "weather": get_weather,
    "daylight": get_daylight,
    "parc_allbrands": get_parc_allbrands
}


def get_exog(name=None, fields=False, data_path=None):
    """
    Discover or fetch exogenous data sources.

    Pass '?' as name for interactive help.

    Args:
        name:      Source identifier. One of:
                     - None              -> list available groups
                     - '?'               -> print usage help
                     - '<group>'         -> return the full CSV as a DataFrame
                     - '<group>#<field>' -> return a single-column DataFrame
        fields:    If True, return the column names for the given group instead
                   of the DataFrame. Ignored when name is None or contains '#'.
        data_path: Optional path to the DATA directory. Overrides auto-detection.

    Returns:
        List of group name strings (when name is None),
        list of column name strings (when fields=True),
        or a DataFrame.

    Examples:
        get_exog('?')                           # print help
        get_exog()                              # list groups
        get_exog('macros')                      # full macros DataFrame
        get_exog('macros', fields=True)         # list field names in macros
        get_exog('macros#bank_lending_rate')    # single-field DataFrame
        get_exog('suzuki_vehicle_sales_annual_2008_2013')
    """
    if name == '?':
        groups = list(_SOURCES)
        width = max(len(g) for g in groups)
        lines = ['Exogenous data groups:', '']
        for g in groups:
            lines.append(f'  {g:<{width}}')
        lines += [
            '',
            'Usage:',
            "  get_exog('?')                    # this help",
            "  get_exog()                       # list group names",
            "  get_exog('<group>')              # full DataFrame",
            "  get_exog('<group>', fields=True) # list column names",
            "  get_exog('<group>#<field>')      # single-column DataFrame",
        ]
        print('\n'.join(lines))
        return

    if name is None:
        return list(_SOURCES)

    if '#' in str(name):
        group, field = name.split('#', 1)
        df = get_exog(group, data_path=data_path)
        if field not in df.columns:
            raise ValueError(
                f"Unknown field {field!r} in group {group!r}. "
                f"Available: {list(df.columns)}"
            )
        return df[[field]]

    if name not in _SOURCES:
        raise ValueError(f"Unknown source {name!r}. Available: {list(_SOURCES)}")

    fn = _SOURCES[name]
    df = fn(data_path) if data_path else fn()

    if fields:
        return list(df.columns)
    return df
def list_exog_groups():
    """
    List all available top-level exogenous data groups.

    Returns:
        List of group name strings that can be passed to get_exog() or list_exog_fields().

    Example:
        list_exog_groups()
        # ['macros', 'suzuki_vehicle_sales_monthly_post_2014', ...]
    """
    return list(_SOURCES)


def list_exog_fields(group, data_path=None):
    """
    List all column names available within a data group.

    Loads and caches the group DataFrame on first call. Useful for exploring
    what fields are available before fetching a specific one.

    Args:
        group:     Group name as returned by list_exog_groups().
        data_path: Optional path to the DATA directory. Overrides auto-detection.

    Returns:
        List of column name strings. Each can be fetched via get_exog('<group>#<field>').

    Example:
        list_exog_fields('macros')
        # ['bank_lending_rate', 'building_permits', 'cpi_transportation', ...]
    """
    df = get_exog(group, data_path=data_path)
    return list(df.columns)
