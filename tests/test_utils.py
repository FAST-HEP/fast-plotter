import pytest
import numpy as np
import pandas as pd
from fast_plotter import utils


binned_df_name = "tests/tbl_datset.njet.csv"


def test_decipher_filename():
    bins, names = utils.decipher_filename("tests/tbl_dataset.njet.csv")
    assert len(names) == 0
    assert set(bins) == set(["dataset", "njet"])

    bins, names = utils.decipher_filename("tests/tbl_dataset.njet--weights.csv")
    assert set(names) == set(["weights"])
    assert set(bins) == set(["dataset", "njet"])

    bins, names = utils.decipher_filename("tests/tbl_njet--weights.csv")
    assert set(names) == set(["weights"])
    assert set(bins) == set(["njet"])

    bins, names = utils.decipher_filename("tests/tbl_njet.csv")
    assert set(names) == set()
    assert set(bins) == set(["njet"])


@pytest.fixture
def binned_df_dataset_njet():
    datasets = ["data", "mc_1", "mc_2"]
    njet = list(range(4))
    index = pd.MultiIndex.from_product((datasets, njet), names=["dataset", "njet"])
    n = np.random.randint(0, 10, len(index))
    sumw = n * np.random.rand()
    sumw2 = sumw * sumw
    df = pd.DataFrame({"n": n, "sumw": sumw, "sumw2": sumw2}, index=index)
    return df


def test_calculate_error(binned_df_dataset_njet):
    df_relative = utils.calculate_error(binned_df_dataset_njet, inplace=False)
    assert np.allclose(df_relative.err,
                       np.true_divide(df_relative.sumw, np.sqrt(df_relative.n)), equal_nan=True)

    df_poisson = utils.calculate_error(binned_df_dataset_njet, inplace=False, do_rel_err=False)
    assert np.allclose(df_poisson.err, np.sqrt(df_poisson.sumw2), equal_nan=True)


def test_read_binned_df():
    df = utils.read_binned_df(binned_df_name)
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names[0] == "dataset"
    assert df.index.names[1] == "njet"
    assert set(df.index.unique(level='dataset')) == set(("data", "mc_1", "mc_2"))
    assert len(df) == 12


def test_drop_over_underflow():
    x1 = np.concatenate(([-np.inf], np.linspace(0, 100, 3), [np.inf]), axis=0)
    x2 = ["one", "TWO", "3"]
    x3 = [10, 11, 20]

    def build_df(*indices):
        index = pd.MultiIndex.from_product(indices, names=list(map(str, range(len(indices)))))
        df = pd.DataFrame({"A": np.arange(len(index))}, index=index)
        return df

    df = build_df(x1)
    cleaned = utils.drop_over_underflow(df)
    assert len(cleaned) == 3
    assert np.array_equal(cleaned.A, np.arange(1, 4))

    df = build_df(x2)
    cleaned = utils.drop_over_underflow(df)
    assert len(cleaned) == 3
    assert np.array_equal(cleaned.A, np.arange(0, 3))

    df = build_df(x3)
    cleaned = utils.drop_over_underflow(df)
    assert len(cleaned) == 3
    assert np.array_equal(cleaned.A, np.arange(0, 3))

    df = build_df(x2, x3, x1)
    cleaned = utils.drop_over_underflow(df)
    assert len(cleaned) == 27
    expected = [(i + 1, i + 4) for i in range(0, 5 * 3 * 3, 5)]
    expected = np.concatenate([np.arange(i, j) for i, j in expected], axis=0)
    assert np.array_equal(cleaned.A, expected)
