from . import utils as utils
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


def plot_all(df, project_1d=True, project_2d=True, data="data", dataset_col="dataset", yscale="log", scale_sims=None):
    figures = {}

    dimensions = utils.binning_vars(df)

    if len(dimensions) == 1:
        figures[(("yscale", yscale),)] = plot_1d(df, yscale=yscale)

    if dataset_col in dimensions:
        dimensions = tuple(dim for dim in dimensions if dim != dataset_col)

    if project_1d and len(dimensions) >= 1:
        for dim in dimensions:
            projected = df.groupby(level=(dim, dataset_col)).sum()
            plot = plot_1d_many(projected, data=data, dataset_col=dataset_col, yscale=yscale, scale_sims=scale_sims)
            figures[(("project", dim), ("yscale", yscale))] = plot

    if project_2d and len(dimensions) > 2:
        logger.warn("project_2d is not yet implemented")

    return figures


def plot_1d_many(df, data="data", dataset_col="dataset", plot_sims="stack", plot_data="sum",
                 yscale="linear", kind_data="scatter", kind_sims="fill", scale_sims=None, summary="ratio"):
    df = utils.convert_intervals(df, to="mid")
    in_df_data, in_df_sims = utils.split_data_sims(df, data_labels=data, dataset_level=dataset_col)
    if scale_sims is not None:
        in_df_sims *= scale_sims

    df_sims = _merge_datasets(in_df_sims, plot_sims, dataset_col, param_name="plot_sims")
    df_data = _merge_datasets(in_df_data, plot_data, dataset_col, param_name="plot_data")

    x_axis = [col for col in df.index.names if col != dataset_col]
    if len(x_axis) > 1:
        raise RuntimeError("Too many dimensions to plot things in 1D")
    if len(x_axis) == 0:
        raise RuntimeError("Too few dimensions to multiple 1D graphs, use plot_1d instead")
    x_axis = x_axis[0]

    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": (3, 1)})
    main_ax, summary_ax = ax

    def _actually_plot(df, kind, label, ax):
        if kind == "scatter":
            df.reset_index().plot.scatter(x=x_axis, y="sumw", yerr="err", color="k", label=label, ax=ax)
        elif kind == "line":
            df["sumw"].unstack(dataset_col).plot.line(drawstyle="steps-mid", ax=ax)
        elif kind == "fill":
            def fill_coll(col, **kwargs):
                ax.fill_between(x=col.index.values, y1=col.values, label=col.name, **kwargs)
            df["sumw"].unstack(dataset_col).apply(fill_coll, axis=0, step="mid")
        else:
            raise RuntimeError("Unknown value for 'kind', '{}'".format(kind))

    _actually_plot(df_sims, kind=kind_sims, label="Monte Carlo", ax=main_ax)
    _actually_plot(df_data, kind=kind_data, label="Data", ax=main_ax)

    main_ax.grid(True)
    main_ax.set_yscale(yscale)
    main_ax.legend()

    if summary == "ratio":
        summed_data = _merge_datasets(in_df_data, "sum", dataset_col=dataset_col)
        summed_sims = _merge_datasets(in_df_sims, "sum", dataset_col=dataset_col)
        plot_ratio(x_axis, summed_data, summed_sims, ax=summary_ax)
    else:
        raise RuntimeError("Unknown value for summary, '{}'".format(kind_data))
    return fig



def _merge_datasets(df, style, dataset_col, param_name="_merge_datasets"):
    if style == "stack":
        utils.calculate_error(df)
        df = utils.stack_datasets(df, dataset_level=dataset_col)
    elif style == "sum":
        df = utils.sum_over_datasets(df, dataset_level=dataset_col)
        utils.calculate_error(df)
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


def plot_ratio(x_axis, data, sims, ax):
    ratio = data.copy()
    s, s_err_sq = sims.sumw, sims.sumw2
    d, d_err_sq = data.sumw, data.sumw2
    s_sq = s * s
    d_sq = d * d
    ratio["Data / MC"] = d / s
    ratio["err"] = (((1 - 2 * d / s) * d_err_sq + d_sq * s_err_sq / s_sq) / s_sq).abs()
    ratio.reset_index().plot.scatter(x=x_axis, y="Data / MC", yerr="err", ax=ax)
    ax.set_ylim([0., 2])
    ax.grid(True)
