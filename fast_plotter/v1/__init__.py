""" Features for version 1 """
from typing import Any, Dict, List
import uproot

from .hist_collections import EfficiencyHistCollection
from .settings import LabelSettings, LegendSettings, GridSettings, TickSettings

from .config import load_config, PlotConfig

def create_collection(name, config):
    LOOKUP = {
        "efficiency": EfficiencyHistCollection,
    }
    collection_class = LOOKUP[config["type"]]
#     return collection_class(**config)
    return collection_class(
        name=name,
        config=config,
    )


def _workaround_uproot_issue38():
    # workaround for issue reading TEfficiency
    # https://github.com/scikit-hep/uproot5/issues/38
    import skhep_testdata
    with uproot.open(skhep_testdata.data_path("uproot-issue38c.root")) as fp:
        hist = fp["TEfficiencyName"]
    # now all TEfficiency objects should be readable
    return hist


def read_histogram_file(input_file, histname):
    with uproot.open(input_file) as fp:
        hist = fp[histname]
        # TODO: use filter_dict to get > 1 hist
    return hist


def make_plots(plot_config_file: str, input_files: List[str], output_dir: str):
    _workaround_uproot_issue38()

    plot_config = load_config(plot_config_file)
    input_file = input_files[0]

    styles = plot_config.pop("styles", {})
    collections = plot_config.pop("collections", {})

    for name, config in collections.items():
        collection = create_collection(name, config)
        sources = config.pop("sources")
        colors = []
        for source in sources:
            label = source.pop("label")
            path = source.pop("path")
            hist = read_histogram_file(input_file, path)
            collection.add_hist(
                name=label,
                numerator=hist.members["fPassedHistogram"].to_numpy()[0],
                denominator=hist.members["fTotalHistogram"].to_numpy()[0],
            )
            colors.append(source.pop("color"))
        collection.hist_colors = colors
        collection.plot()
        collection.save(output_dir)
