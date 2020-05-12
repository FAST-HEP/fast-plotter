import pandas as pd
from fast_curator import read
from .functions import multiply_dataframe


def scale_datasets(df, curator_cfg, multiply_by=[], divide_by=[], dataset_col="dataset"):
    if isinstance(curator_cfg, list):
        dataset_cfg = curator_cfg
    else:
        dataset_cfg = read.from_yaml(curator_cfg)

    scale = [1] * len(dataset_cfg)
    for dataset in dataset_cfg:
        for m in multiply_by:
            scale *= getattr(dataset, m)
        for d in divide_by:
            scale /= getattr(dataset, d)

    scale = pd.Series(scale, index=[d.name for d in dataset_cfg], name=dataset_col)
    return multiply_dataframe(df, scale)
