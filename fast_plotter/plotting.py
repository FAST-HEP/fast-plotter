from . import utils as utils
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


def plot_all(df, project_1d=True, project_2d=True, data="data", dataset_col="dataset", yscale="log", scale_sims=None):
    figures = {}

    dimensions = utils.binning_vars(df)

    if len(dimensions) == 1:
        figures[(None, None )] = plot_1d(df, yscale=yscale)

    if dataset_col in dimensions:
        dimensions = tuple(dim for dim in dimensions if dim != dataset_col)

    if project_1d and len(dimensions) >= 1:
        for dim in dimensions:
            projected = df.groupby(level=(dim, dataset_col)).sum()
            utils.calculate_error(projected)
            figures[(("project", dim),)] = plot_1d_many(projected, data=data, dataset_col=dataset_col, yscale=yscale)

    if project_2d and len(dimensions) > 2:
        logger.warn("project_2d is not yet implemented")

    return figures


def plot_1d_many(df, data="data", dataset_col="dataset", plot_sims="stack", plot_data="sum", yscale="linear", kind_data="scatter", kind_sims="line", scale_sims=None):
    df = utils.convert_intervals(df, to="mid")
    df_data, df_sims = utils.split_data_sims(df, data_labels=data, dataset_level=dataset_col)
    if scale_sims is not None:
        df_sims = scale_sims * df_sims.copy()

    df_sims = _merge_datasets(df_sims, plot_sims, "plot_sims", dataset_col)
    df_data = _merge_datasets(df_data, plot_data, "plot_data", dataset_col)

    x_axis = [col for col in df.index.names if col != dataset_col]
    if len(x_axis) > 1:
        raise RuntimeError("Too many dimensions to plot things in 1D")
    if len(x_axis) == 0:
        raise RuntimeError("Too few dimensions to multiple 1D graphs, use plot_1d instead")
    x_axis = x_axis[0]

    fig, ax = plt.subplots(1)

    if kind_sims == "scatter":
        df_sims.reset_index().plot.scatter(x=x_axis, y="sumw", yerr="err", color="k", label="Monte Carlo", ax=ax)
    elif kind_sims == "line":
        df_sims["sumw"].unstack(dataset_col).plot.line(drawstyle="steps-mid", ax=ax)
    else:
        raise RuntimeError("Unknown value for kind_sims, '{}'".format(kind_sims))

    if kind_data == "scatter":
        df_data.reset_index().plot.scatter(x=x_axis, y="sumw", yerr="err", color="k", label="data", ax=ax)
    elif kind_data == "line":
        df_data["sumw"].unstack(dataset_col).plot.line(drawstyle="steps-mid", ax=ax)
    else:
        raise RuntimeError("Unknown value for kind_data, '{}'".format(kind_data))

    ax.grid(True)
    ax.set_yscale(yscale)
    return fig



def _merge_datasets(df, style, param_name, dataset_col):
    if style == "stack":
        df = utils.stack_datasets(df, dataset_level=dataset_col)
    elif style == "sum":
        df = utils.sum_over_datasets(df, dataset_level=dataset_col)
    elif style:
        msg = "'{}' must be either 'sum', 'stack' or None. Got {}"
        raise RuntimeError(msg.format(param_name, style))
    return df


def plot_1d(df, kind="line", yscale="lin"):
    fig, ax = plt.subplots(1)
    df["sumw"].plot(kind=kind)
    plt.grid(True)
    plt.yscale(yscale)
    return fig
