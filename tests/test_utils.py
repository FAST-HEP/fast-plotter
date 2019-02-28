import pytest
import numpy as np
import pandas as pd
from fast_plotter import utils


@pytest.fixture
def binned_df_dataset_njet():
    datasets = ["data", "mc_1", "mc_2"]
    njet = list(range(4))
    index = pd.MultiIndex.from_product((datasets, njet))
    n = np.random.randint(0, 10, len(index))
    sumw = n * np.random.rand()
    sumw2 = sumw * sumw
    df = pd.DataFrame({"n": n, "sumw": sumw, "sumw2": sumw2}, index=index)
    return df


def test_calculate_error(binned_df_dataset_njet):
    utils.calculate_error(binned_df_dataset_njet)
    assert all(binned_df_dataset_njet.err == np.sqrt(binned_df_dataset_njet.sumw2))
