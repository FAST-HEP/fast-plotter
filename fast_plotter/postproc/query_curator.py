import pandas as pd
from fast_curator import read


def _get_cfg(cfg):
    if isinstance(cfg, list):
        return cfg
    return read.from_yaml(cfg)


def prepare_datasets_scale_factor(curator_cfg, multiply_by=[], divide_by=[], dataset_col="dataset", eventtype="mc"):
    dataset_cfg = _get_cfg(curator_cfg)

    sfs = {}
    for dataset in dataset_cfg:
        if eventtype and dataset.eventtype not in eventtype:
            sfs[dataset.name] = 1
            continue

        scale = 1
        for m in multiply_by:
            scale *= float(getattr(dataset, m))
        for d in divide_by:
            scale /= float(getattr(dataset, d))
        sfs[dataset.name] = scale

    sfs = pd.Series(sfs, name=dataset_col)
    return sfs


def make_dataset_map(curator_cfg, map_from="name", map_to="eventtype",
                     default_from=None, default_to=None, error_all_missing=True):
    dataset_cfg = _get_cfg(curator_cfg)

    mapping = {}
    missing_from = 0
    missing_to = 0
    for dataset in dataset_cfg:
        if hasattr(dataset, map_from):
            key = getattr(dataset, map_from)
        else:
            key = default_from
            missing_from += 1

        if hasattr(dataset, map_to):
            value = getattr(dataset, map_to)
        else:
            value = default_to
            missing_to += 1

        mapping[key] = value
    if missing_from == len(dataset_cfg) and error_all_missing:
        msg = "None of the datasets contain the 'from' field, '%s'"
        raise RuntimeError(msg % map_from)

    if missing_to == len(dataset_cfg) and error_all_missing:
        msg = "None of the datasets contain the 'to' field, '%s'"
        raise RuntimeError(msg % map_to)

    return mapping
