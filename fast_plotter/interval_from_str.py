import pandas as pd


_interval_regex = r"^(?P<open>[[(])"
_interval_regex += r"(?P<low>-inf|[-+]?[0-9][.0-9]*)"
_interval_regex += r"\s*,\s*"
_interval_regex += r"(?P<high>\+?inf|[-+]?[0-9][.0-9]*)"
_interval_regex += r"(?P<close>[)\]])$"


def interval_from_string(series):
    if not pd.api.types.is_string_dtype(series):
        return series
    extracted = series.str.extract(_interval_regex)
    extracted = extracted.dropna()
    if len(extracted) != len(series):
        return series
    left_closed = extracted.open.unique()
    right_closed = extracted.close.unique()
    if len(left_closed) != 1 or len(right_closed) != 1:
        return series
    left_closed = left_closed[0] == "["
    right_closed = right_closed[0] == "]"
    if left_closed:
        if right_closed:
            closed = "both"
        else:
            closed = "left"
    else:
        if right_closed:
            closed = "right"
        else:
            closed = "neither"

    left = pd.to_numeric(extracted.low)
    right = pd.to_numeric(extracted.high)
    interval = pd.IntervalIndex.from_arrays(left=left,
                                            right=right,
                                            closed=closed)
    return interval


def convert_intervals(df, to="mid", level=[], column=[]):
    df = convert_intervals_level(df, to=to, select=level)
    df = convert_intervals_column(df, to=to, select=column)
    return df


def convert_intervals_level(df, to="mid", select=[]):
    def _convert_interval_index(index, to):
        if isinstance(index, pd.IntervalIndex):
            converted = getattr(index, to)
            converted.name = index.name
            return converted
        return index

    if not isinstance(df.index, pd.MultiIndex):
        df.index = _convert_interval_index(df.index, to)
        return df

    for name, level in zip(df.index.names, df.index.levels):
        if select and name not in select:
            continue
        out_level = _convert_interval_index(level, to)
        df.index = df.index.set_levels([out_level], [name])
    return df


def convert_intervals_column(df, to="mid", select=[]):
    for col in df.columns:
        if select and col not in select:
            continue
        if hasattr(df[col], to):
            df[col] = getattr(df[col], to)
    return df
