import pytest
import numpy as np
import pandas as pd
from fast_plotter import utils


binned_df_name = "tests/tbl_datset.njet.csv"


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
    utils.calculate_error(binned_df_dataset_njet)
    assert all(binned_df_dataset_njet.err == np.sqrt(binned_df_dataset_njet.sumw2))


def test_read_binned_df():
    df = utils.read_binned_df(binned_df_name)
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names[0] == "dataset"
    assert df.index.names[1] == "njet"
    assert set(df.index.unique(level='dataset')) == set(("data", "mc_1", "mc_2"))
    assert len(df) == 12
