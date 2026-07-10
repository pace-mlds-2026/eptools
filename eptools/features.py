import os
import pandas as pd
from eptools.data_preprocessing import _DATAFRAMES_CACHE, _resolve_data_path, _freeze


def _load_source(relative_path, cache_key, date_format=None, data_path=None,
                 numeric_cols=None) -> pd.DataFrame:
    """
    Generic cached CSV loader for feature data sources.

    Resolves the data path, checks the in-memory cache, reads the CSV,
    parses the 'date' column as a DatetimeIndex, freezes the result, and
    returns a mutable copy.

    Args:
        relative_path: Path to the CSV relative to the resolved DATA directory.
        cache_key:     Unique string used as the in-memory cache key.
        date_format:   Optional strptime format string passed to pd.to_datetime.
                       Omit for standard ISO / mixed date strings.
        data_path:     Optional override for the DATA directory.
        numeric_cols:  Optional list of column names to coerce to numeric
                       via pd.to_numeric(errors='coerce'). Use to clean
                       stray non-numeric tokens (e.g. "ALL NaN") that would
                       otherwise force a column to object dtype. Columns not
                       listed (incl. categorical ones) are left untouched.

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
    # Coerce only the named columns, so a stray "ALL NaN" becomes real NaN
    # instead of forcing the column to object dtype. Leaves everything else
    # (incl. legitimately categorical columns) untouched.
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    _DATAFRAMES_CACHE[key] = _freeze(df)
    return _DATAFRAMES_CACHE[key].copy()



def get_suzuki_vehicle_sales_monthly_post_2014(data_path=None) -> pd.DataFrame:
    """Load ANAC Suzuki monthly vehicle sales data (2014+)."""
    return _load_source(
        "API_SOURCES/API_SOURCE_vehicle_sales_monthly_2014_onwards.csv",
        "__anac_vehicle_sales__",
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
        date_format="%d/%m/%Y",
        data_path=data_path,
        numeric_cols=["housing_index"],
    )

def get_suzuki_claims(data_path=None) -> pd.DataFrame:
    """Load monthly Suzuki insurance claims data (2020–2024)."""
    return _load_source(
        "API_SOURCES/API_SOURCE_suzuki_claims.csv",
        "__suzuki_claims__",
        data_path=data_path,
    )


def get_suzuki_fleet_size(data_path=None) -> pd.DataFrame:
    """Load monthly Suzuki active fleet size data (2014+)."""
    return _load_source(
        "API_SOURCES/API_SOURCE_suzuki_fleet_size.csv",
        "__suzuki_fleet_size__",
        data_path=data_path,
    )


_SOURCES = {
    "macros": get_macros,
    "suzuki_vehicle_sales_monthly_post_2014": get_suzuki_vehicle_sales_monthly_post_2014,
    "weather": get_weather,
    "daylight": get_daylight,
    "parc_allbrands": get_parc_allbrands,
    "suzuki_claims": get_suzuki_claims,
    "suzuki_fleet_size": get_suzuki_fleet_size
}



def get_feature(name=None, fields=False, data_path=None):
    """
    Discover or fetch feature data sources.

    Pass '?' as name for interactive help.

    Args:
        name:      Source identifier. One of:
                     - None              -> list available groups
                     - '?'               -> print usage help
                     - '<group>'         -> return the full CSV as a DataFrame
                     - '<group>#<field>' -> return a single-column DataFrame
        fields:    If True, return fully-qualified '<group>#<field>' strings for
                   every column in the group. Each string is directly usable as
                   the `name` argument to get_feature() or add_feature().
                   Ignored when name is None or contains '#'.
        data_path: Optional path to the DATA directory. Overrides auto-detection.

    Returns:
        List of group name strings (when name is None),
        list of '<group>#<field>' strings (when fields=True),
        or a DataFrame.

    Examples:
        get_feature('?')                            # print help
        get_feature()                               # list groups
        get_feature('macros')                       # full macros DataFrame
        get_feature('macros', fields=True)          # ['macros#bank_lending_rate', ...]
        get_feature('macros#bank_lending_rate')     # single-field DataFrame
        get_feature('suzuki_vehicle_sales_annual_2008_2013')
    """
    if name == '?':
        groups = list(_SOURCES)
        width = max(len(g) for g in groups)
        lines = ['Feature data groups:', '']
        for g in groups:
            lines.append(f'  {g:<{width}}')
        lines += [
            '',
            'Usage:',
            "  get_feature('?')                     # this help",
            "  get_feature()                        # list group names",
            "  get_feature('<group>')               # full DataFrame",
            "  get_feature('<group>', fields=True)  # list '<group>#<field>' names",
            "  get_feature('<group>#<field>')       # single-column DataFrame",
        ]
        print('\n'.join(lines))
        return

    if name is None:
        return list(_SOURCES)

    if '#' in str(name):
        group, field = name.split('#', 1)
        df = get_feature(group, data_path=data_path)
        if field not in df.columns:
            raise ValueError(
                f"Unknown field {field!r} in group {group!r}. "
                f"Available: {get_feature(group, fields=True)}"
            )
        return df[[field]]

    if name not in _SOURCES:
        raise ValueError(f"Unknown source {name!r}. Available: {list(_SOURCES)}")

    fn = _SOURCES[name]
    df = fn(data_path) if data_path else fn()

    if fields:
        return [f"{name}#{col}" for col in df.columns]
    return df



def list_feature_groups():
    """
    List all available top-level feature data groups.

    Returns:
        List of group name strings that can be passed to get_feature() or
        list_feature_fields().

    Example:
        list_feature_groups()
        # ['macros', 'suzuki_vehicle_sales_monthly_post_2014', ...]
    """
    return list(_SOURCES)


def list_feature_fields(group, data_path=None):
    """
    List all fully-qualified field names available within a data group.

    Returns '<group>#<field>' strings that can be passed directly to
    get_feature() or add_feature().

    Args:
        group:     Group name as returned by list_feature_groups().
        data_path: Optional path to the DATA directory.

    Returns:
        List of '<group>#<field>' strings.

    Example:
        list_feature_fields('macros')
        # ['macros#bank_lending_rate', 'macros#building_permits', ...]
    """
    return get_feature(group, fields=True, data_path=data_path)



def add_feature(df, name, data_path=None):
    """
    Add a feature column to a collision-sales-format DataFrame.

    Fetches the feature data identified by `name` (same syntax as get_feature)
    and left-joins it onto `df` using the feature 'date' index aligned to
    the 'month' column of `df`.

    Args:
        df:        DataFrame in get_collision_sales_df() format, with a 'month'
                   column of month-start datetimes.
        name:      Source identifier passed to get_feature(). Supports the same
                   '<group>' and '<group>#<field>' syntax.
        data_path: Optional path to the DATA directory. Passed through to
                   get_feature().

    Returns:
        A copy of `df` with the feature column(s) appended (left join on 'month').

    Examples:
        add_feature(df, 'macros#bank_lending_rate')
        add_feature(df, 'daylight')
        add_feature(df, 'suzuki_vehicle_sales_monthly_post_2014#total_units')
    """
    feature = get_feature(name, data_path=data_path)
    result = (
        df.merge(feature.reset_index(), left_on="month", right_on="date", how="left")
          .drop(columns=["date"])
    )
    null_cols = [c for c in feature.columns if result[c].isna().all()]
    if null_cols:
        df_dates  = df["month"].dropna().head(3).tolist()
        feat_dates = feature.reset_index()["date"].dropna().head(3).tolist()
        raise ValueError(
            f"add_feature({name!r}): join produced no matches — "
            f"all values for {null_cols} are NaN.\n"
            f"  'month' sample (left):  {df_dates}\n"
            f"  'date'  sample (right): {feat_dates}"
        )
    return result



list_feature_groups()
