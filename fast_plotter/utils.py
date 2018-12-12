import re
import os
import numpy as np
import pandas as pd
from .interval_from_str import interval_from_string, convert_intervals


def decipher_filename(filename):
    decipher = re.compile(r"tbl_(?P<binning>.*?)(?P<weights>--.*|)\.csv")
    groups = decipher.match(os.path.basename(filename))

    binning = groups.group("binning").split(".")
    weights = groups.group("weights").split(".")
    return binning, weights

def get_read_options(filename):
    index_cols, _ = decipher_filename(filename)
    options = dict(index_col=range(len(index_cols)),
                   comment="#"
                   )
    return options


def read_binned_df(filename):
    read_opts = get_read_options(filename)
    df = pd.read_csv(filename, **read_opts)
    columns = df.index.names[:]
    df.reset_index(inplace=True)
    for col in df.columns:
        df[col] = interval_from_string(df[col])
    df.set_index(columns, inplace=True)
    return df


def binning_vars(df):
    if isinstance(df.index, pd.MultiIndex):
        return tuple(df.index.names)
    return (df.index,)


def weighting_vars(df):
    weight_vars = []
    for name in df.columns:
        weight = name.split(":")[0]
        if weight not in weight_vars:
            weight_vars.append(weight)

    return tuple(weight_vars)


def split_df(df, first_values, level=0):
    if df is None:
        return None, None
    if isinstance(first_values, str):
        regex = re.compile(first_values)
        first_values = [val for val in df.index.unique(level) if regex.match(val)]
    second = df.drop(first_values, level=level)
    second_values = second.index.unique(level=level)
    first = df.drop(second_values, level=level)
    if len(first) == 0:
        first = None
    if len(second) == 0:
        second = None
    return first, second


def split_data_sims(df, data_labels=["data"], dataset_level="dataset"):
    return split_df(df, first_values=data_labels, level=dataset_level)


def calculate_error(df, sumw2_label="sumw2", err_label="err", inplace=True):
    if not inplace:
        df = df.copy()
    for column in df:
        if sumw2_label in column:
            err_name = column.replace(sumw2_label, err_label)
            df[err_name] = np.sqrt(df[column])
    if not inplace:
        return df


def groupby_all_but(df, by=None, level=None):
    if level is None and by is None:
        raise RuntimeError("Either 'by' or 'level' must be provided")
    if level is not None and by is not None:
        raise RuntimeError("Only one of 'by' or 'level' should be provided, not both")

    args = {}
    if level:
        if not isinstance(df.index, pd.MultiIndex):
            raise RuntimeError("Cannot use 'level' to groupby for non-multiindex DataFrame")
        if not isinstance(level, (tuple, list)):
            level = (level,)
        
        group_levels = [l for l in df.index.names if l not in level]
        args["level"] = group_levels

    if by:
        if not isinstance(by, (tuple, list)):
            by = (by,)
        group_columns = [l for l in df.columns if l not in by]
        args["by"] = group_columns

    return df.groupby(**args)


def stack_datasets(df, dataset_level="dataset"):
    groups = groupby_all_but(df, level=dataset_level)
    stacked = groups.cumsum()
    return stacked


def sum_over_datasets(df, dataset_level="dataset"):
    groups = groupby_all_but(df, level=dataset_level)
    summed = groups.sum()
    return summed
