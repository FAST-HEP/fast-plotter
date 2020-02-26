import re
import six
import os
import numpy as np
import pandas as pd
from .interval_from_str import interval_from_string, convert_intervals


__all__ = ["interval_from_string", "convert_intervals"]


def decipher_filename(filename):
    decipher = re.compile(r"tbl_(?P<binning>.*?)(--(?P<names>.*)|)\.csv")
    groups = decipher.match(os.path.basename(filename))

    binning = groups.group("binning").split(".")
    name = []
    if groups.group("names"):
        name = groups.group("names").split(".")
    return binning, name


def get_read_options(filename):
    index_cols, _ = decipher_filename(filename)
    options = dict(index_col=list(range(len(index_cols))),
                   comment="#"
                   )
    return options


def read_binned_df(filename, **kwargs):
    read_opts = get_read_options(filename)
    read_opts.update(kwargs)
    df = pd.read_csv(filename, **read_opts)
    columns = df.index.names[:]
    df.reset_index(inplace=True)
    dtype = kwargs.pop("dtype", [])
    for col in df.columns:
        if col in dtype:
            continue
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


def mask_rows(df, regex, level=None):
    if isinstance(regex, six.string_types):
        regex = re.compile(regex)
    index = df.index
    if level is not None:
        index = index.get_level_values(level)
    data_rows = index.map(lambda x: bool(regex.search(str(x))))
    return data_rows


def split_df(df, first_values, level=0):
    if df is None:
        return None, None
    if isinstance(first_values, six.string_types):
        regex = re.compile(first_values)
        first_values = [val for val in df.index.unique(level) if regex.match(val)]
    if not first_values:
        return None, df
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


def calculate_error(df, sumw2_label="sumw2", err_label="err", inplace=True, do_rel_err=True):
    if not inplace:
        df = df.copy()
    if do_rel_err:
        root_n = np.sqrt(df["n"])
    for column in df:
        if do_rel_err and column.endswith("sumw"):
            err_name = column.replace("sumw", err_label)
            errs = np.true_divide(df[column], root_n)
            errs[~np.isfinite(errs)] = np.nan
            df[err_name] = errs
        elif not do_rel_err and sumw2_label in column:
            err_name = column.replace(sumw2_label, err_label)
            df[err_name] = np.sqrt(df[column])
    if not inplace:
        return df


def groupby_all_but(df, by=None, level=None):
    if level is None and by is None:
        raise RuntimeError("Either 'by' or 'level' must be provided")
    if level is not None and by is not None:
        raise RuntimeError(
            "Only one of 'by' or 'level' should be provided, not both")

    args = {}
    if level:
        if not isinstance(df.index, pd.MultiIndex):
            raise RuntimeError(
                "Cannot use 'level' to groupby for non-multiindex DataFrame")
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


def order_datasets(df, dataset_order, dataset_level="dataset", values="sumw"):
    if isinstance(dataset_order, str) and dataset_order.startswith("sum"):
        sums = df.groupby(level=dataset_level).sum()
        if dataset_order == "sum-ascending":
            dataset_order = sums.sort_values(
                by=values, ascending=True).index.tolist()
        elif dataset_order == "sum-descending":
            dataset_order = sums.sort_values(
                by=values, ascending=False).index.tolist()
        return df.reindex(dataset_order, axis=0, level=dataset_level)
    elif isinstance(dataset_order, list):
        return df.reindex(dataset_order, axis=0, level=dataset_level)
    raise RuntimeError("Bad dataset_order value")


def rename_index(df, name_replacements):
    if not isinstance(df.index, pd.MultiIndex):
        return df
    df.index.names = [name_replacements.get(n, n) for n in df.index.names]
    return df


def drop_over_underflow(df):
    index = df.index.to_frame()
    index = index.select_dtypes(exclude=['object'])
    good_rows = np.isfinite(index).all(axis=1)
    return df.loc[good_rows]
