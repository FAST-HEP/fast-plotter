import pytest
import string
import numpy as np
import pandas as pd
from fast_plotter.postproc import functions as funcs


def _make_string(index):
    chars = string.printable
    start = index % len(chars)
    stop = (index + 28) % len(chars)
    return chars[start] + chars[stop]


@pytest.fixture
def binned_df():
    anInt = list(range(4))
    aCat = ["foo", "bar"]
    anInterval = pd.IntervalIndex.from_breaks(np.linspace(100, 104, 6))
    index = pd.MultiIndex.from_product([anInt, aCat, anInterval],
                                       names=["int", "cat", "interval"])
    data = dict(a=np.arange(len(index)), b=map(_make_string, range(len(index))))
    df = pd.DataFrame(data, index=index)
    return df


def test_query(binned_df):
    result = funcs.query(binned_df, "cat=='bar'")
    assert len(result) == 20

    result = funcs.query(binned_df, "cat=='bar' and int > 2")
    assert len(result) == 5


def test_rebin(binned_df):
    result = funcs.rebin(binned_df.copy(), rename="hruff", axis="int", mapping=dict(zip(range(4), [0, 2] * 2)))
    assert len(result) == 20
    assert list(result.index.unique("hruff")) == [0, 2]

    mapping = {0: dict(bar="foo"), 2: dict(foo="bar"), 3: dict(foo="BAZ", bar="BAZ")}
    result = funcs.rebin(binned_df.copy(), axis=["int", 'cat'], mapping=mapping)
    assert len(result) == 25
    assert set(result.index.unique("cat")) == {"bar", "BAZ", "foo"}


def test_keep_bins(binned_df):
    result = funcs.keep_bins(binned_df.copy(), "int", keep=[0, 2])
    assert len(result) == 20

    result = funcs.keep_bins(binned_df.copy(), "int", keep=binned_df.index.unique("int"))
    assert len(result) == 40


def test_keep_specific_bins(binned_df):
    keep = {"0": ["foo"], "1": ["bar"]}
    result = funcs.keep_specific_bins(binned_df, axis=["int", "cat"], keep=keep)
    assert len(result) == 10

    result = funcs.keep_specific_bins(binned_df, axis=["int", "cat"], keep=keep, expansions=[])
    assert len(result) == 10

    expansions = dict(one=["bar"])
    keep = {"0": ["foo"], "1": ["{one}"]}
    result = funcs.keep_specific_bins(binned_df, axis=["int", "cat"], keep=keep, expansions=expansions)
    assert len(result) == 10

    expansions = dict(one=["bar"], two=["foo", "bar"])
    keep = {"0": ["{two}"], "1": ["{one}"]}
    result = funcs.keep_specific_bins(binned_df, axis=["int", "cat"], keep=keep, expansions=expansions)
    assert len(result) == 15


def test_combine_cols_AND_split_dimension(binned_df):
    result = funcs.combine_cols(binned_df, {"a;b": "{a};{b}"})
    assert len(result.columns) == 3
    assert all(result.columns == ["a", "b", "a;b"])
    assert len(result) == 40

# def test_split_dimension(binned_df):
#     #def split_dimension(df, axis, delimeter=";"):
#     result = funcs.split_dimension(binned_df, ["interval"], ",")
#     pass

# def test_regex_split_dimension():
#     #def regex_split_dimension(df, axis, regex):
#     pass

# def test_rename_cols():
#     #def rename_cols(df, mapping):
#     pass


def test_rename_dim(binned_df):
    result = funcs.rename_dim(binned_df, {"int": "integers", "cat": "CATEGORICALS"})
    assert result.index.names == ["integers", "CATEGORICALS", "interval"]


def test_split(binned_df):
    results = funcs.split(binned_df, "cat", keep_split_dim=True)
    assert len(results) == 2
    assert all([r[0].index.nlevels == 3 for r in results])

    results = funcs.split(binned_df, "int", keep_split_dim=True)
    assert len(results) == 4
    assert all([r[0].index.nlevels == 3 for r in results])


def test_filter_cols(binned_df):
    df = binned_df.index.to_frame()

    result = funcs.filter_cols(df, items=["int"])
    assert len(result.columns) == 1
    assert result.columns[0] == "int"

    result = funcs.filter_cols(df, items=["int", "cat"])
    assert len(result.columns) == 2
    assert set(result.columns) == set(("int", "cat"))

    result = funcs.filter_cols(df, like="int")
    assert len(result.columns) == 2
    assert set(result.columns) == set(("int", "interval"))

    result = funcs.filter_cols(df, like=["int", "cat"])
    assert len(result.columns) == 3
    assert set(result.columns) == set(("int", "cat", "interval"))

    result = funcs.filter_cols(df, regex="^int.*")
    assert len(result.columns) == 2
    assert set(result.columns) == set(("int", "interval"))


# def test_reorder_dimensions():
#     #def reorder_dimensions(df, order):
#     pass

# def test_densify():
#     #def densify(df, known={}):
#     pass

# def test_stack_weights():
#     #def stack_weights(df, drop_n_col=False):
#     pass

# def test_to_datacard_inputs():
#     #def to_datacard_inputs(df, select_data, rename_syst_vars=False):
#     pass

# def test_assign_col():
#     #def assign_col(df, assignments={}, evals={}, drop_cols=[]):
#     pass

# def test_assign_dim():
#     #def assign_dim(df, assignments={}, evals={}, drop_cols=[]):
#     pass

# def test_merge():
#     #def merge(dfs):
#     pass

def test_multiply_values(binned_df):
    result = funcs.multiply_values(binned_df, constant=3)
    assert np.array_equal(result.a, np.arange(len(binned_df)) * 3)
    assert np.array_equal(result.b, binned_df.b)

    result = funcs.multiply_values(binned_df, apply_if="int % 2 == 0", constant=3)
    assert np.array_equal(result.a, np.arange(len(binned_df)) * np.repeat([3, 1, 3, 1], 10))
    assert np.array_equal(result.b, binned_df.b)

    result = funcs.multiply_values(binned_df, mapping={"int % 2 == 0": 3, "int % 2 == 1": 7.2})
    assert np.array_equal(result.a, np.arange(len(binned_df)) * np.repeat([3, 7.2, 3, 7.2], 10))
    assert np.array_equal(result.b, binned_df.b)

    result = funcs.multiply_values(binned_df, mapping={"cat=='foo'": 1.2, "cat=='bar'": 19})
    tiled_vals = np.tile(np.repeat([1.2, 19], 5), 4)
    assert np.array_equal(result.a, np.arange(len(binned_df)) * tiled_vals)
    assert np.array_equal(result.b, binned_df.b)


# def test_multiply_dataframe():
#     #def multiply_dataframe(df, multiply_df, use_column=None):
#     pass

# def test_normalise_group():
#     #def normalise_group(df, groupby_dimensions, apply_if=None, use_column=None):
#     pass

# def test_open_many():
#     #def open_many(file_list, return_meta=True):
#     pass

# def test_write_out():
#     #def write_out(df, meta, filename="tbl_{dims}--{name}.csv", out_dir=None):
#     pass
