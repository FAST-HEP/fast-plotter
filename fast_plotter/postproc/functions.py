import os
import six
import re
import numpy as np
import pandas as pd
from .query_curator import prepare_datasets_scale_factor, make_dataset_map
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class BinningDimCombiner():
    def __init__(self, index, cols_to_merge, combine_dims_ignore=None, delimiter="__"):
        col_names = index.names
        self.col_names = col_names
        self.cols_to_merge = [col_names.index(col) for col in cols_to_merge]
        self.cols_not_merged = [i for i, name in enumerate(col_names) if name not in cols_to_merge]
        self.ignore = combine_dims_ignore
        self.delimiter = delimiter

    def __call__(self, row):
        unmerged = tuple([row[i] for i in self.cols_not_merged])
        if self.ignore is not None:
            merged = [str(row[i]) for i in self.cols_to_merge if row[i] != self.ignore]
        else:
            merged = [str(row[i]) for i in self.cols_to_merge]
        result = (self.delimiter.join(merged), ) + unmerged
        return result


def handle_one_df(df, query=None, replacements=[],
                  combine_dims=[], combine_dims_ignore=None, combine_delim="__", engine_query=None):
    if query:
        df.query(query, inplace=True, engine=engine_query)
    if df.empty:
        return
    df.drop(df.filter(like="Unnamed").columns, axis=1, inplace=True)
    for column, repl in replacements:
        if column not in df.index.names:
            continue
        df.rename(repl, level=column, inplace=True, axis="index")
    df = df.groupby(level=df.index.names).sum()
    if combine_dims:
        names = df.index.names
        combiner = BinningDimCombiner(df.index, combine_dims,
                                      combine_dims_ignore,
                                      delimiter=combine_delim)
        new_index = df.index.to_frame().apply(combiner, axis="columns", result_type="expand")
        df.index = pd.MultiIndex.from_frame(new_index)
        df.index.names = combiner(names)
    return df


def query(df, query, engine=None):
    """
    Keep only rows that satisfy requirements of the query string,
    See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
    """
    logger.info("Applying query: %s", query)
    return handle_one_df(df, query=query, engine_query=engine)


def rebin(df, axis, mapping, ignore_when_combining=None, rename=None, drop_others=False):
    """
    Rename and / or collect bins and categories together
    """
    logger.info("Rebinning on axis, '%s'", axis)
    if isinstance(axis, (int, float, six.string_types)):
        replacements = [(axis, mapping)]
        out = handle_one_df(df, replacements=replacements)
        if drop_others:
            out = keep_bins(out, axis, set(mapping.values()))
        if rename is not None:
            out.index.set_names(rename, level=axis, inplace=True)
        return out

    out_df = handle_one_df(df, combine_dims=axis, combine_delim=";")

    def explode(mapping, expect_depth, prefix="", depth=0):
        exploded_map = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                result = explode(value, expect_depth, str(key) + ";", depth + 1)
                exploded_map.update(result)
            else:
                exploded_map[prefix + key] = prefix + value
        return exploded_map
    exploded_map = explode(mapping, len(axis))
    replacements = [(";".join(axis), exploded_map)]
    out_df = handle_one_df(out_df, replacements=replacements)

    if drop_others:
        out_df = keep_bins(out_df, axis, set(exploded_map.values()))

    if rename is None:
        out_df = split_dimension(out_df, axis)
    elif isinstance(rename, list) and len(rename) == len(axis):
        out_df = split_dimension(out_df, axis)
        out_df.index.set_names(rename, level=axis, inplace=True)
    else:
        out_df.index.set_names(rename, level=replacements[0], inplace=True)

    return out_df


def rebin_by_curator_cfg(df, curator_cfg, map_from="name", map_to="eventtype",
                         column_from="dataset", column_to=None,
                         default_from=None, default_to=None, error_all_missing=True):
    mapping = make_dataset_map(curator_cfg,
                               map_from=map_from, map_to=map_to,
                               default_from=default_from,
                               default_to=default_to,
                               error_all_missing=error_all_missing)
    df = rebin(df, axis=column_from, mapping=mapping, rename=column_to)
    return df


def split_dimension(df, axis, delimeter=";"):
    """
    Split up a binning dimensions
    """
    logger.info("Splitting on axis, '%s'", axis)
    combined_name = delimeter.join(axis)
    index = df.index.to_frame()
    split_index = index[combined_name].str.split(delimeter, expand=True)
    for i, col in enumerate(axis):
        index[col] = split_index[i]
    index.drop(combined_name, axis="columns", inplace=True)
    df.set_index(pd.MultiIndex.from_frame(index), inplace=True, drop=True)
    return df


def keep_bins(df, axis, keep):
    """Keep bins on the single dimension, dropping others"""
    others = {val for val in df.index.unique(axis) if val not in keep}
    if not others:
        return df
    logger.info("Dropping values for '%s': %s", axis, str(others))
    out = df.drop(others, level=axis, axis="index")
    return out


def keep_specific_bins(df, axis, keep, expansions={}):
    """
    Keep all the specified bins and drop the others
    """
    logger.info("Keeping values based on '%s'", str(axis))
    if not isinstance(axis, list):
        out_df = keep_bins(df, axis, keep)
        keep = [k.format(**{name: v}) for name, vals in expansions.items() for v in vals for k in keep]
        return out_df

    delim = ";"
    out_df = handle_one_df(df, combine_dims=axis, combine_delim=delim)
    expanded = [(tuple(), keep)]
    for ax in axis[:-1]:
        expanded = [(last + (new, ), remaining) for last, new_items in expanded for new, remaining in new_items.items()]
    for ax in axis[-1:]:
        expanded = [keys + (new, ) for keys, vals in expanded for new in vals]
        expanded = [delim.join(vals) for vals in expanded]

    if expansions:
        replacements = [dict()]
        for name, vals in expansions.items():
            replacements = [{name: v, **r} for v in vals for r in replacements]
        expanded = [e.format(**r) for r in replacements for e in expanded]
    out_df = keep_bins(out_df, delim.join(axis), expanded)
    out_df = split_dimension(out_df, axis, delimeter=delim)
    return out_df


def filter_cols(df, items=None, like=None, regex=None, drop_not_keep=False):
    """Filter out columns you want to keep.

    Parameters:
      items (list-like): A list of column names to filter with
      like (str, list[string]): A string or list of strings which will filter
            columns where they are found in the column name
      regex (str): A regular expression to match column names to
      drop_not_keep (bool): Inverts the selection if true so that matched columns are dropped
    """
    if not like or not isinstance(like, (tuple, list)):
        df_filtered = df.filter(items=items, like=like, regex=regex)
    elif like:
        if items and like:
            raise RuntimeError("Can only use one of 'items', 'like', or 'regex'")
        filtered = [set(col for col in df.columns if i in col) for i in like]
        filtered = set.union(*filtered)
        df_filtered = df.filter(items=filtered, regex=regex)

    if drop_not_keep:
        return df.drop(df_filtered.columns)
    return df_filtered


def combine_cols(df, format_strings, as_index=[]):
    """Combine columns together using format strings"""
    logger.info("Combining columns based on: %s", str(format_strings))
    result_names = list(format_strings.keys())

    def apply_fmt(row):
        return [s.format(**row) for s in format_strings.values()]

    index = df.index.names
    new_df = df.reset_index()
    results = new_df.apply(apply_fmt, axis="columns", result_type="expand")
    results.columns = result_names
    new_df = new_df.assign(**results)
    new_df.set_index(index, inplace=True, drop=True)
    if as_index:
        new_df.set_index(as_index, inplace=True, append=True)
    return new_df


def regex_split_dimension(df, axis, regex):
    """
    Split up a binning dimensions using a regex and pandas.Series.str.extract
    """
    logger.info("Splitting on axis, '%s' with regex '%s'", axis, regex)
    index = df.index.to_frame()
    split_index = index[axis].str.extract(regex, expand=True)
    index.drop(axis, axis="columns", inplace=True)
    for col in split_index.columns:
        index[col] = split_index[col]
    df.set_index(pd.MultiIndex.from_frame(index), inplace=True, drop=True)
    return df


def rename_cols(df, mapping):
    """Rename one or more value columns"""
    df.rename(mapping, inplace=True, axis="columns")
    return df


def rename_dim(df, mapping):
    """
    Rename one or more dimensions
    """
    df.index.names = [mapping.get(n, n) for n in df.index.names]
    return df


def split(df, axis, keep_split_dim, return_meta=True):
    """
    split the dataframe into a list of dataframes using a given binning
    dimensions
    """
    def to_tuple(obj):
        if isinstance(obj, (list, tuple)):
            return tuple(obj)
        else:
            return (obj, )

    axis = to_tuple(axis)
    logger.info("Splitting on axis: '%s'", axis)
    out_dfs = []
    groups = df.groupby(level=axis, group_keys=keep_split_dim)
    for split_val, group in groups:
        split_val = to_tuple(split_val)
        if not keep_split_dim:
            group.index = group.index.droplevel(axis)
        result = group.copy()
        if return_meta:
            meta = dict(zip(axis, split_val))
            split_name = "--".join(["{}_{}".format(*i) for i in meta.items()])
            meta["split_name"] = split_name
            result = (result, meta)
        out_dfs.append(result)
    return out_dfs


def reorder_dimensions(df, order):
    """
    Reorder the binning dimensions
    """
    logger.info("Ordering binning dimensions according to: %s", str(order))
    return df.reorder_levels(order, axis="index")


def densify(df, known={}):
    """
    Densify axes with known values and / or values from axes on other
    dimensions
    """
    logger.info("Densify dimensions")
    before_len = len(df)

    index_vals = []
    names = df.index.names
    for dim in range(df.index.nlevels):
        vals = df.index.unique(dim)
        _known = known.get(df.index.names[dim], None)
        if _known is None:
            _known = known.get(dim, None)
        if _known is not None:
            vals = vals.union(_known)
        index_vals.append(vals)

    df = df.reindex(pd.MultiIndex.from_product(index_vals, names=names), axis="index")

    after_len = len(df)
    logger.info("    Change to number of bins (after - before): %d", after_len - before_len)
    return df


def stack_weights(df, drop_n_col=False):
    """
    Convert from a wide-form to long-form binned dataframe
    """
    logger.info("Stacking weights")
    if drop_n_col:
        df.drop("n", inplace=True, axis="columns")
    else:
        df.set_index("n", append=True, inplace=True)
    df.columns = pd.MultiIndex.from_tuples([c.split(":") for c in df.columns], names=["systematic", ""])
    out = df.stack(0, dropna=False)
    if not drop_n_col:
        out.reset_index(level="n", inplace=True)
    return out


def to_datacard_inputs(df, select_data, rename_syst_vars=False, error_from_n=False):
    """
    Convert to a long-form dataframe suitable as input to fast-datacard
    """
    logger.info("Converting to datacard inputs")
    if rename_syst_vars:
        df.columns = [n.replace("_up:", "_Up:").replace("_down:", "_Down:") for n in df.columns]
    df = stack_weights(df)
    data_mask = df.eval(select_data)
    df["content"] = df.n
    df["content"][~data_mask] = df.sumw
    if error_from_n:
        df["error"] = df.content / np.sqrt(df.n)
    else:
        df["error"] = np.sqrt(df.sumw2)
    df.drop(["n", "sumw", "sumw2"], inplace=True, axis="columns")
    return df


def generic_pandas(df, func, *args, **kwargs):
    """
    Apply generic pandas function to each input
    """
    logger.info("Apply generic pandas function")
    return getattr(df, func)(*args, **kwargs)


def unstack_weights(df, weight_name="systematic", includes_counts=True):
    """
    The inverse to stack_weights
    """
    logger.info("Unstacking systematics")
    if includes_counts:
        df = df.set_index("n", append=True)
    df = df.unstack(weight_name)
    df.columns = ["{1}:{0}".format(*c) if c[1] else c[0] for c in df.columns]
    if includes_counts:
        df = df.reset_index(level="n")
    return df


def assign_col(df, assignments={}, evals={}, drop_cols=[]):
    """
    Add or change columns by assigning or evaluating new ones
    """
    for key, value in evals.items():
        df[key] = df.eval(value)

    if assignments:
        df = df.assign(**assignments)

    if drop_cols:
        df.drop(drop_cols, axis="columns", inplace=True)
    return df


def assign_dim(df, assignments={}, evals={}, drop_cols=[]):
    """
    Add binning dimensions to the index by assigning or evaluating
    """
    index = df.index.to_frame()
    index = assign_col(index, assignments=assignments, evals=evals, drop_cols=drop_cols)
    df.set_index(pd.MultiIndex.from_frame(index), inplace=True, drop=True)
    return df


def merge(dfs, sort=True):
    """ Merge a list of binned dataframes together
    """
    logger.info("Merging %d dataframes", len(dfs))
    final_df = pd.concat(dfs, sort=sort)  # .fillna(float("-inf"))
    final_df = final_df.groupby(level=final_df.index.names, sort=sort).sum()  # .replace(float("-inf"), float("nan"))
    return final_df


def multiply_values(df, constant=0, mapping={}, weight_by_dataframes=[], apply_if=None):
    logger.info("Multiplying values")
    mask = slice(None)
    if apply_if:
        mask = df.eval(apply_if)
        if mask.dtype.kind != "b":
            msg = "'apply_if' statement doesn't return a boolean: %s"
            raise ValueError(msg % apply_if)
        index = df.index
        ignored = df.loc[~mask]
        df = df.loc[mask]
    if mapping:
        for select, value in mapping.items():
            df = multiply_values(df, constant=value, apply_if=select)
    if weight_by_dataframes:
        for mul_df in weight_by_dataframes:
            df = multiply_dataframe(df, mul_df)
    if constant:
        numeric_cols = df.select_dtypes('number')
        df = df.assign(**numeric_cols.multiply(constant))
    if apply_if:
        df = pd.concat([df, ignored]).reindex(index=index)

    return df


def multiply_dataframe(df, multiply_df, use_column=None, level=None):
    if isinstance(multiply_df, six.string_types):
        multiply_df = open_many([multiply_df], return_meta=False)[0]
    if use_column is not None:
        multiply_df = multiply_df[use_column]
    if isinstance(multiply_df, pd.Series):
        out = df.mul(multiply_df, axis=0, level=level)
    else:
        out = df.mul(multiply_df, level=level)
    return out


def scale_datasets(df, curator_cfg, multiply_by=[], divide_by=[],
                   dataset_col="dataset", eventtype="mc", use_column=None):
    """
    Pull fields from a fast-curator config for datasets, and use these to normalise inputs
    """
    scale = prepare_datasets_scale_factor(curator_cfg, multiply_by, divide_by, dataset_col, eventtype)
    result = multiply_dataframe(df, scale, use_column=use_column, level=dataset_col)
    return result


def normalise_group(df, groupby_dimensions, apply_if=None, use_column=None):
    logger.info("Normalising within groups defined by: %s", str(groupby_dimensions))
    norm_to = 1 / df.groupby(level=groupby_dimensions).sum()
    normed = multiply_dataframe(df, multiply_df=norm_to, use_column=use_column)
    return normed


def open_many(file_list, value_columns=r"(.*sumw2?|^n$)", return_meta=True):
    """ Open a list of dataframe files
    """
    dfs = []
    for fname in file_list:
        df = pd.read_csv(fname, comment='#')
        drop_cols = [col for col in df.columns if "Unnamed" in col]
        df.drop(drop_cols, inplace=True, axis="columns")
        index_cols = [col for col in df.columns if not re.match(value_columns, col)]
        df.set_index(index_cols, inplace=True, drop=True)
        if return_meta:
            name = fname
            if "--" in name:
                name = name.split("--")[-1]
                name = os.path.splitext(name)[0]
            df = (df, dict(name=name, filename=fname))
        dfs.append(df)
    return dfs


def write_out(df, meta, filename="tbl_{dims}--{name}", out_dir=None, filetype="csv"):
    """ Write a dataframe to disk
    """
    meta = meta.copy()
    meta["dims"] = ".".join(df.index.names)

    complete_file = filename.format(**meta)
    if out_dir:
        complete_file = os.path.join(out_dir, complete_file)
    os.makedirs(os.path.dirname(complete_file), exist_ok=True)
    logger.info("Writing out file '%s'", complete_file)
    if not complete_file.endswith(filetype):
        complete_file += "." + filetype
    if filetype == "csv":
        df.to_csv(complete_file)
    elif filetype == "hd5":
        df.to_hdf(complete_file, key="df")
    return df
