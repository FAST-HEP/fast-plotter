import os
import six
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
                  combine_dims=[], combine_dims_ignore=None, combine_delim="__"):
        if query:
            df.query(query, inplace=True)
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


class BaseManipulator():
    def __init__(self, **kwargs):
        use_outdir = getattr(self, "use_outdir", False)
        if not use_outdir:
            kwargs.pop("out_dir", None)
        self.name = kwargs.pop("name")
        self.kwargs = kwargs
        self.func = globals()[self.func]

    def __call__(self, dfs):
        if self.cardinality == "many-to-one":
            out = [self.func(dfs, **self.kwargs)]
        elif self.cardinality == "one-to-many":
            out = []
            for df in dfs:
                out += self.func(df, **self.kwargs)
        elif self.cardinality == "one-to-one":
            out = [self.func(df, **self.kwargs) for df in dfs]
        elif self.cardinality == "none-to-many":
            out = dfs + self.func(**self.kwargs)
        return out


class LabelledDataframe:
    def __init__(self, df, name=None, meta=None):
        self.meta = meta.copy() if meta is not None else {}

        if name is not None:
            if "--" in name:
                name = name.split("--")[-1]
                name = os.path.splitext(name)[0]
            self.meta["name"] = name
        self.meta["dims"] = df.index.names
        self.df = df

    def insert(self, key, value):
        self.meta[key] = value

    def __getitem__(self, key):
        return self.meta[key]

    @property
    def name(self):
        return self["name"]


def query(df, query):
    """
    Keep only rows that satisfy requirements of the query string,
    See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
    """
    logger.info("Applying query: %s", query)
    out = handle_one_df(df.df, query=query)
    return LabelledDataframe(out, meta=df.meta)


class Query(BaseManipulator):
    cardinality = "one-to-one"
    func = "query"


def rebin(df, axis, mapping, ignore_when_combining=None, rename=None, drop_others=False):
    """
    Rename and / or collect bins and categories together
    """
    logger.info("Rebinning on axis, '%s'", axis)
    if isinstance(axis, (int, float, six.string_types)):
        replacements = [(axis, mapping)]
        out = handle_one_df(df.df, replacements=replacements)
        if drop_others:
            out = keep_bins(out, axis, set(mapping.values()))
        if rename is not None:
            out.index.set_names(rename, level=axis, inplace=True)
        return LabelledDataframe(out, meta=df.meta)

    out_df = handle_one_df(df.df, combine_dims=axis, combine_delim=";")

    def explode(mapping, expect_depth, prefix="", depth=0):
        exploded_map = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                result = explode(value, expect_depth, key + ";", depth +1)
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

    out_df = LabelledDataframe(out_df, meta=df.meta)
    return out_df


class ReBin(BaseManipulator):
    cardinality = "one-to-one"
    func = "rebin"


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
        return LabelledDataframe(out_df, meta=df.meta)

    delim = ";"
    out_df = handle_one_df(df.df, combine_dims=axis, combine_delim=delim)
    expanded = [(tuple(), keep)]
    for ax in axis[:-1]:
        expanded = [(last + (new, ), remaining) for last, new_items in expanded for new, remaining in new_items.items()]
    for ax in axis[-1:]:
        expanded = [keys + (new, ) for keys, vals in expanded for new in vals]
        expanded = [delim.join(vals) for vals in expanded]

    expanded = [k.format(**{name: v}) for name, vals in expansions.items() for v in vals for k in expanded]
    out_df = keep_bins(out_df, delim.join(axis), expanded)
    out_df = split_dimension(out_df, axis, delimeter=delim)
    return LabelledDataframe(out_df, meta=df.meta)


class KeepSpecificBins(BaseManipulator):
    cardinality = "one-to-one"
    func = "keep_specific_bins"


def combine_cols(df, format_strings):
    """Combine columns together using format strings"""
    logger.info("Combining columns based on: %s", str(format_strings))
    result_names = list(format_strings.keys())
    def apply_fmt(row):
       return [s.format(**row) for s in format_strings.values()]
    index = df.df.index.names
    new_df = df.df.reset_index()
    results = new_df.apply(apply_fmt, axis="columns", result_type="expand")
    results.columns = result_names
    new_df = new_df.assign(**results)
    new_df.set_index(index, inplace=True, drop=True)
    return LabelledDataframe(new_df, meta=df.meta)


class CombineColumns(BaseManipulator):
    cardinality = "one-to-one"
    func = "combine_cols"


def regex_split_dimension(df, axis, regex):
    """
    Split up a binning dimensions using a regex and pandas.Series.str.extract
    """
    logger.info("Splitting on axis, '%s' with regex '%s'", axis, regex)
    index = df.df.index.to_frame()
    split_index = index[axis].str.extract(regex, expand=True)
    index.drop(axis, axis="columns", inplace=True)
    for col in split_index.columns:
        index[col] = split_index[col]
    df.df.set_index(pd.MultiIndex.from_frame(index), inplace=True, drop=True)
    return LabelledDataframe(df.df, meta=df.meta)


class RegexSplitDimension(BaseManipulator):
    cardinality = "one-to-one"
    func = "regex_split_dimension"


def rename_cols(df, mapping):
    """Rename one or more value columns"""
    df.df.rename(mapping, inplace=True, axis="columns")
    return df


class RenameCols(BaseManipulator):
    cardinality = "one-to-one"
    func = "rename_cols"


def rename_dim(df, mapping):
    """
    Rename one or more dimensions
    """
    df.df.index.names = [mapping.get(n, n) for n in df.df.index.names]
    return LabelledDataframe(df.df, meta=df.meta)


class RenameBinningDimension(BaseManipulator):
    cardinality = "one-to-one"
    func = "rename_dim"


def split(df, axis, keep_split_dim):
    """
    split the dataframe into a list of dataframes using a given binning
    dimensions
    """
    logger.info("Splitting on axis: '%s'", axis)
    out_dfs = []
    groups = df.df.groupby(level=axis, group_keys=keep_split_dim)
    for split_val, group in groups:
        if not keep_split_dim:
            group.index = group.index.droplevel(split_on_dimension)
        out = LabelledDataframe(group.copy(), meta=df.meta)
        out.insert("split_name", "%s_%s" % (axis, split_val))
        out.insert(axis, split_val)
        out_dfs.append(out)
    return out_dfs


class Split(BaseManipulator):
    cardinality = "one-to-many"
    func = "split"


def reorder_dimensions(df, order):
    """
    Reorder the binning dimensions 
    """
    logger.info("Ordering binning dimensions according to: %s", str(order))
    out = df.df.reorder_levels(order, axis="index")
    return LabelledDataframe(out, meta=df.meta)


class ReorderDimensions(BaseManipulator):
    cardinality = "one-to-one"
    func = "reorder_dimensions"


def densify(in_df, known={}):
    """
    Densify axes with known values and / or values from axes on other
    dimensions
    """
    logger.info("Densify dimensions")
    df = in_df.df
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
    in_df.df = df
    return in_df


class Densify(BaseManipulator):
    cardinality = "one-to-one"
    func = "densify"


def stack_weights(df, drop_n_col=False):
    """
    Convert from a wide-form to long-form binned dataframe
    """
    logger.info("Stacking weights")
    if drop_n_col:
        df.df.drop("n", inplace=True, axis="columns")
    df.df.columns = pd.MultiIndex.from_tuples([c.split(":") for c in df.df.columns], names=["systematic", ""])
    out = df.df.stack(0, dropna=False)
    return LabelledDataframe(out, meta=df.meta)


class StackWeights(BaseManipulator):
    cardinality = "one-to-one"
    func = "stack_weights"


def to_datacard_inputs(df, select_data, rename_syst_vars=False):
    """
    Convert to a long-form dataframe suitable as input to fast-datacard
    """
    meta = df.meta
    logger.info("Converting to datacard inputs")
    if rename_syst_vars:
        df.df.columns = [n.replace("_up:", "_Up:").replace("_down:", "_Down:") for n in df.df.columns]
    df.df.set_index("n", append=True, inplace=True)
    df = stack_weights(df).df
    df.reset_index(level="n", inplace=True)
    data_mask = df.eval(select_data)
    df["content"] = df.n
    df["content"][~data_mask] = df.sumw
    df["error"] = df.content / np.sqrt(df.n)

    df.drop(["n", "sumw", "sumw2"], inplace=True, axis="columns")
    return LabelledDataframe(df, meta=meta)


class ToDatacardInputs(BaseManipulator):
    cardinality = "one-to-one"
    func = "to_datacard_inputs"


def assign_col(df, assignments={}, evals={}, drop_cols=[]):
    """
    Add or change columns by assigning or evaluating new ones
    """
    for key, value in evals.items():
        df.df[key] = df.df.eval(value)

    if assignments:
        df.df = df.df.assign(**assignments)

    if drop_cols:
        df.df.drop(drop_cols, axis="columns", inplace=True)
    return LabelledDataframe(df.df, meta=df.meta)


class AssignCol(BaseManipulator):
    cardinality = "one-to-one"
    func = "assign_col"


def assign_dim(df, assignments={}, evals={}, drop_cols=[]):
    """
    Add binning dimensions to the index by assigning or evaluating
    """
    index = LabelledDataframe(df.df.index.to_frame(), name="index")
    index = assign_col(index, assignments=assignments, evals=evals, drop_cols=drop_cols).df
    df.df.set_index(pd.MultiIndex.from_frame(index), inplace=True, drop=True)
    return LabelledDataframe(df.df, name=df.name)


class AssignDim(BaseManipulator):
    cardinality = "one-to-one"
    func = "assign_dim"


def merge(in_dfs, keys=None):
    """ Merge a list of binned dataframes together
    """
    logger.info("Merging %d dataframes", len(in_dfs))
    assert len(set(df["name"] for df in in_dfs)) == 1
    dfs = [df.df for df in in_dfs]
    final_df = pd.concat(dfs, sort=True)#.fillna(float("-inf"))
    final_df = final_df.groupby(level=final_df.index.names).sum()#.replace(float("-inf"), float("nan"))
    return LabelledDataframe(final_df, name=in_dfs[0]["name"])


class Merge(BaseManipulator):
    cardinality = "many-to-one"
    func = "merge"


def multiply_values(df, constant=0, mapping={}, weight_by_dataframes=[], apply_if=None):
    logger.info("Multiplying values")
    in_df = df.df
    mask = slice(None)
    if apply_if:
        mask = in_df.eval(apply_if)
        if mask.dtype.kind != "b":
            msg = "'apply_if' statement doesn't return a boolean: %s"
            raise ValueError(msg % apply_if)
        in_df = in_df.loc[mask]
    if constant:
        out_df = in_df * constant
    if mapping:
        raise NotImplementedError("'mapping' option not yet implemented")
    if weight_by_dataframes:
        out_df = in_df
        for mul_df in weight_by_dataframes:
            out_df = multiply_dataframe(out_df, mul_df)
    if apply_if:
        out_df = pd.concat([out_df, in_df.loc[~mask]])
    return LabelledDataframe(out_df, meta=df.meta)


class MultiplyValues(BaseManipulator):
    cardinality = "one-to-one"
    func = "multiply_values"


def multiply_dataframe(df, multiply_df, use_column=None):
    if isinstance(multiply_df, six.string_types):
        multiply_df = open_many([multiply_df])[0]
    print("BEK BEFORE", df.df.head())
    print("BEK muply", multiply_df.head())
    if use_column is not None:
        multiply_df = multiply_df[use_column]
    if isinstance(multiply_df, pd.Series):
        out = df.df.mul(multiply_df, axis=0)
    else:
        out = df.df * multiply_df
    print("BEK after", out.head())
    return LabelledDataframe(out, df.meta)


def normalise_group(df, groupby_dimensions, apply_if=None, use_column=None):
    logger.info("Normalising within groups defined by: %s", str(groupby_dimensions))
    norm_to = 1 / df.df.groupby(level=groupby_dimensions).sum()
    normed = multiply_dataframe(df, multiply_df=norm_to, use_column=use_column).df
    return LabelledDataframe(normed, meta=df.meta)


class NormaliseGroup(BaseManipulator):
    cardinality = "one-to-one"
    func = "normalise_group"


def open_many(file_list):
    """ Open a list of dataframe files
    """
    dfs = []
    for fname in file_list:
        df = pd.read_csv(fname, comment='#')
        drop_cols = [col for col in df.columns if "Unnamed" in col]
        df.drop(drop_cols, inplace=True, axis="columns")
        index_cols = [col for col in df.columns if "sumw" not in col and col != "n"]
        df.set_index(index_cols, inplace=True, drop=True)
        df = LabelledDataframe(df, name=fname)
        dfs.append(df)
    return dfs


class OpenMany(BaseManipulator):
    cardinality = "none-to-many"
    func = "open_many"


def write_out(df, filename="tbl_{dims}--{name}.csv", out_dir=None):
    """ Write a dataframe to disk
    """
    dims = ".".join(df["dims"])
    meta = df.meta.copy()
    meta["dims"] = dims
    complete_file = filename.format(**meta)
    if out_dir:
        complete_file = os.path.join(out_dir, complete_file)
    os.makedirs(os.path.dirname(complete_file), exist_ok=True)
    logger.info("Writing out file '%s'", complete_file)
    df.df.to_csv(complete_file)
    return df


class WriteOut(BaseManipulator):
    cardinality = "one-to-one"
    func = "write_out"
    use_outdir = True
