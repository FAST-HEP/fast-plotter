from . import utils as utils
import traceback
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


def plot_all(df, project_1d=True, project_2d=True, data="data", signal=None, dataset_col="dataset",
             yscale="log", lumi=None, annotations=[], dataset_order="sum-ascending",
             bin_variable_replacements={}, **kwargs):
    figures = {}

    dimensions = utils.binning_vars(df)

    if len(dimensions) == 1:
        df = utils.rename_index(df, bin_variable_replacements)
        figures[(("yscale", yscale),)] = plot_1d(
            df, yscale=yscale, annotations=annotations)

    if dataset_col in dimensions:
        dimensions = tuple(dim for dim in dimensions if dim != dataset_col)

    if project_1d and len(dimensions) >= 1:
        for dim in dimensions:
            logger.info("Making 1D Projection: " + dim)
            projected = df.groupby(level=(dim, dataset_col)).sum()
            projected = utils.rename_index(
                projected, bin_variable_replacements)
            if dataset_order is not None:
                projected = utils.order_datasets(
                    projected, dataset_order, dataset_col)
            try:
                plot = plot_1d_many(projected, data=data, signal=signal,
                                    dataset_col=dataset_col, scale_sims=lumi)
                figures[(("project", dim), ("yscale", yscale))] = plot
            except Exception as e:
                logger.error("Couldn't plot 1D projection: " + dim)
                logger.error(traceback.print_exc())
                logger.error(e)

    if project_2d and len(dimensions) > 2:
        logger.warn("project_2d is not yet implemented")

    return figures


class fill_coll(object):
    def __init__(self, n_colors=10, ax=None, fill=True, line=True):
        self.calls = 0
        colormap = plt.cm.nipy_spectral
        self.colors = [colormap(i)
                       for i in np.linspace(.96, .2, n_colors)]
        self.ax = ax
        self.fill = fill
        self.line = line

    def pre_call(self, col):
        ax = self.ax
        if not ax:
            ax = plt.gca()
        color = self.colors[self.calls]
        x = col.index.values
        y = col.values
        return ax, x, y, color

    def __call__(self, col, **kwargs):
        ax, x, y, color = self.pre_call(col)
        if x.dtype.kind in 'biufc':
            x, y = pad_zero(x, y)
        if self.fill:
            ax.fill_between(x=x, y1=y, label=col.name,
                            linewidth=0, color=color, **kwargs)
        if self.line:
            if self.fill:
                label = None
                color = "k"
                width = 0.5
            else:
                color = None
                label = col.name
                width = 1
            ax.step(x=x, y=y, color=color, linewidth=width, where="mid", label=label)
        self.calls += 1


class bar_coll(fill_coll):
    def __call__(self, col, **kwargs):
        ax, x, y, color = self.pre_call(col)
        align = "center"
        if x.dtype.kind in 'biufc':
            align = "edge"
        facecolor = list(color)
        facecolor[-1] *= 0.5
        ax.bar(x, y, edgecolor=color, facecolor=facecolor, width=1, label=col.name, align=align)
        self.calls += 1


def actually_plot(df, x_axis, y, yerr, kind, label, ax, dataset_col="dataset"):
    n_datasets = len(df.index.unique(dataset_col))
    if kind == "scatter":
        df.reset_index().plot.scatter(x=x_axis, y=y, yerr=yerr,
                                      color="k", label=label, ax=ax, s=13)
        return
    elif kind == "line":
        filler = fill_coll(n_datasets, ax=ax, fill=False)
        df[y].unstack(dataset_col).iloc[:, ::-1].apply(filler, axis=0, step="mid")
        return
    elif kind == "bar":
        filler = bar_coll(n_datasets, ax=ax)
        df[y].unstack(dataset_col).iloc[:, ::-1].apply(filler, axis=0, step="mid")
    elif kind == "fill":
        filler = fill_coll(n_datasets, ax=ax)
        df[y].unstack(dataset_col).iloc[:, ::-1].apply(filler, axis=0, step="mid")
    elif kind == "fill-error-last":
        actually_plot(df, x_axis, y, yerr, "fill",
                      label, ax, dataset_col=dataset_col)
        summed = df.unstack(dataset_col)
        last_dataset = summed.columns.get_level_values(1)[n_datasets - 1]
        summed = summed.xs(last_dataset, level=1, axis="columns")
        x = summed.index.values
        y_down = summed[y] - summed[yerr]
        y_up = summed[y] + summed[yerr]
        x, y_down, y_up = pad_zero(x, y_down.values, y_up.values)
        ax.fill_between(x=x, y2=y_down, y1=y_up,
                        color="gray", step="mid", alpha=0.7)
    else:
        raise RuntimeError("Unknown value for 'kind', '{}'".format(kind))


def pad_zero(x, *y_values):
    do_pad_left = not np.isneginf(x[0])
    do_pad_right = not np.isposinf(x[-1])
    width_slice = x[None if do_pad_left else 1:None if do_pad_right else -1]
    mean_width = width_slice[0]
    if len(width_slice) > 1:
        mean_width = np.diff(width_slice).mean()
    x_left_padding = [x[0] - mean_width, x[0]
                      ] if do_pad_left else [x[1] - mean_width]
    x_right_padding = [x[-1], x[-1] + mean_width] if do_pad_right else [x[-2] + mean_width]

    x = np.concatenate((x_left_padding, x[1:-1], x_right_padding))
    new_values = []
    for y in y_values:
        y_left_padding = [0, y[1]] if do_pad_left else [0]
        y_right_padding = [y[-2], 0] if do_pad_right else [0]
        y[np.isnan(y)] = 0
        y = np.concatenate((y_left_padding, y[1:-1], y_right_padding))
        new_values.append(y)

    return (x,) + tuple(new_values)


def plot_1d_many(df, prefix="", data="data", signal=None, dataset_col="dataset",
                 plot_sims="stack", plot_data="sum", plot_signal=None,
                 kind_data="scatter", kind_sims="fill-error-last", kind_signal="line",
                 scale_sims=None, summary="ratio"):
    y = "sumw"
    yvar = "sumw2"
    yerr = "err"
    if prefix:
        y = prefix + ":" + y
        yvar = prefix + ":" + yvar
        yerr = prefix + ":" + yerr

    df = utils.convert_intervals(df, to="mid")
    in_df_data, in_df_sims = utils.split_data_sims(
        df, data_labels=data, dataset_level=dataset_col)
    if scale_sims is not None:
        in_df_sims[y] *= scale_sims
        in_df_sims[yvar] *= scale_sims * scale_sims
    if signal:
        in_df_signal, in_df_sims = utils.split_data_sims(
            in_df_sims, data_labels=signal, dataset_level=dataset_col)
    else:
        in_df_signal = None

    if in_df_data is None or in_df_sims is None:
        summary = None
    if not summary:
        fig, main_ax = plt.subplots(1, 1)
    else:
        fig, ax = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
        fig.subplots_adjust(hspace=.1)
        main_ax, summary_ax = ax

    x_axis = [col for col in df.index.names if col != dataset_col]
    if len(x_axis) > 1:
        raise RuntimeError("Too many dimensions to plot things in 1D")
    if len(x_axis) == 0:
        raise RuntimeError(
            "Too few dimensions to multiple 1D graphs, use plot_1d instead")
    x_axis = x_axis[0]

    config = [(in_df_sims, plot_sims, kind_sims, "Monte Carlo", "plot_sims"),
              (in_df_data, plot_data, kind_data, "Data", "plot_data"),
              (in_df_signal, plot_signal, kind_signal, "Signal", "plot_signal"),
              ]
    for df, combine, style, label, var_name in config:
        if df is None or len(df) == 0:
            continue
        merged = _merge_datasets(df, combine, dataset_col, param_name=var_name)
        actually_plot(merged, x_axis=x_axis, y=y, yerr=yerr, kind=style,
                      label=label, ax=main_ax, dataset_col=dataset_col)
    main_ax.set_xlabel(x_axis)

    if not summary:
        return main_ax, None

    summary_ax = ax[1]
    if summary == "ratio":
        main_ax.set_xlabel("")
        summed_data = _merge_datasets(
            in_df_data, "sum", dataset_col=dataset_col)
        summed_sims = _merge_datasets(
            in_df_sims, "sum", dataset_col=dataset_col)
        plot_ratio(summed_data, summed_sims, x=x_axis,
                   y=y, yvar=yvar, ax=summary_ax)
    else:
        raise RuntimeError("Unknown value for summary, '{}'".format(kind_data))
    return main_ax, summary_ax


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


def add_annotations(annotations, ax):
    for cfg in annotations:
        cfg = cfg.copy()
        s = cfg.pop("text")
        xy = cfg.pop("position")
        cfg.setdefault("xycoords", "axes fraction")
        ax.annotate(s, xy=xy, **cfg)


def plot_1d(df, kind="line", yscale="lin"):
    fig, ax = plt.subplots(1)
    df["sumw"].plot(kind=kind)
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.yscale(yscale)
    return fig


def plot_ratio(data, sims, x, y, yvar, ax):
    ratio = data.copy()
    s, s_err_sq = sims[y], sims[yvar]
    d, d_err_sq = data[y], data[yvar]
    s_sq = s * s
    d_sq = d * d
    ratio["Data / MC"] = d / s
    ratio["err"] = (((1 - 2 * d / s) * d_err_sq + d_sq * s_err_sq / s_sq) / s_sq).abs()
    ratio.reset_index().plot.scatter(x=x, y="Data / MC", yerr="err", ax=ax, color="k")
    ax.set_ylim([0., 2])
    ax.grid(True)
    ax.set_axisbelow(True)
