from . import utils as utils
from . import statistics as stats
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import logging
logger = logging.getLogger(__name__)


def change_brightness(color, amount):
    if amount is None:
        return
    import colorsys
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    if isinstance(color, tuple):
        color = mc.to_rgb(c)
    c = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_all(df, project_1d=True, project_2d=True, data="data", signal=None, dataset_col="dataset",
             yscale="log", lumi=None, annotations=[], dataset_order=None,
             continue_errors=True, bin_variable_replacements={}, colourmap="nipy_spectral",
             figsize=None, **kwargs):
    figures = {}

    dimensions = utils.binning_vars(df)
    ran_ok = True

    if len(dimensions) == 1:
        df = utils.rename_index(df, bin_variable_replacements)
        figures[(("yscale", yscale),)] = plot_1d(
            df, yscale=yscale, annotations=annotations)

    if dataset_col in dimensions:
        dimensions = tuple(dim for dim in dimensions if dim != dataset_col)
        if dataset_order is None:
            dataset_order = df.index.unique(dataset_col).tolist()

    if project_1d and len(dimensions) >= 1:
        for dim in dimensions:
            logger.info("Making 1D Projection: " + dim)
            projected = df.groupby(level=(dim, dataset_col)).sum()
            projected = utils.rename_index(
                projected, bin_variable_replacements)
            projected = utils.order_datasets(projected, "sum-ascending", dataset_col)
            try:
                plot = plot_1d_many(projected, data=data, signal=signal,
                                    dataset_col=dataset_col, scale_sims=lumi,
                                    colourmap=colourmap, dataset_order=dataset_order,
                                    figsize=figsize, **kwargs
                                    )
                figures[(("project", dim), ("yscale", yscale))] = plot
            except Exception as e:
                if not continue_errors:
                    raise
                logger.error("Couldn't plot 1D projection: " + dim)
                logger.error(traceback.print_exc())
                logger.error(e)
                ran_ok = False

    if project_2d and len(dimensions) > 2:
        logger.warn("project_2d is not yet implemented")

    return figures, ran_ok


class ColorDict():
    def __init__(self, order=None, named=None, n_colors=10, cmap="nipy_spectral", cmap_start=0.96, cmap_stop=0.2):
        self.order = {}
        if order is not None:
            self.order = {n: i for i, n in enumerate(order)}
            n_colors = max(n_colors, len(order))

        if isinstance(cmap, str):
            colmap_def = plt.get_cmap(cmap)
            n_colors = max(colmap_def.N, n_colors) if colmap_def.N < 256 else n_colors
        elif isinstance(cmap, dict):
            colmap_def = plt.get_cmap(cmap.get("map"))
            n_colors = cmap.get("n_colors", n_colors)
            cmap_start = cmap.get("colour_start", cmap_start)
            cmap_stop = cmap.get("colour_stop", cmap_stop)

        self.defaults = [colmap_def(i) for i in np.linspace(cmap_start, cmap_stop, n_colors)]
        self.named = named if named is not None else {}

    def get_colour(self, index=None, name=None):
        if index is None and name is None:
            raise RuntimeError("'Index' and 'name' cannot both be None")

        if name in self.named:
            return self.named[name]

        if name in self.order:
            return self.defaults[self.order[name]]

        if index is None:
            raise RuntimeError("'index' was not provided and we got an unknown named object '%s'" % name)

        return self.defaults[index]


class FillColl(object):
    def __init__(self, n_colors=10, ax=None, fill=True, line=True, dataset_colours=None,
                 colourmap="nipy_spectral", dataset_order=None, linewidth=0.5, expected_xs=None):
        self.calls = -1
        self.expected_xs = expected_xs
        self.colors = ColorDict(n_colors=n_colors, order=dataset_order,
                                named=dataset_colours, cmap=colourmap)

        self.ax = ax
        self.fill = fill
        self.line = line
        self.linewidth = linewidth

    def pre_call(self, column):
        ax = self.ax
        if not ax:
            ax = plt.gca()
        color = self.colors.get_colour(index=self.calls, name=column.name)
        x = column.index.values
        y = column.values
        return ax, x, y, color

    def __call__(self, col, **kwargs):
        ax, x, y, color = self.pre_call(col)
        if self.fill:
            draw(ax, "fill_between", x=x, ys=["y1"],
                 y1=y, label=col.name, expected_xs=self.expected_xs,
                 linewidth=0, color=color, **kwargs)
        if self.line:
            if self.fill:
                label = None
                color = "k"
                width = self.linewidth
                style = "-"
            else:
                color = None
                label = col.name
                width = 2
                style = "--"
            draw(ax, "step", x=x, ys=["y"], y=y, expected_xs=self.expected_xs,
                 color=color, linewidth=width, where="mid", label=label, linestyle=style)
        self.calls += 1


class BarColl(FillColl):
    def __call__(self, col, **kwargs):
        ax, x, y, color = self.pre_call(col)
        align = "center"
        if x.dtype.kind in 'biufc':
            align = "edge"
        facecolor = list(color)
        facecolor[-1] *= 0.5
        ax.bar(x, y, edgecolor=color, facecolor=facecolor, width=1, label=col.name, align=align)
        self.calls += 1


def actually_plot(df, x_axis, y, yerr, kind, label, ax, dataset_col="dataset",
                  dataset_colours=None, colourmap="nipy_spectral", dataset_order=None):
    expected_xs = df.index.unique(x_axis).values
    if kind == "scatter":
        draw(ax, "errorbar", x=df.reset_index()[x_axis], ys=["y", "yerr"], y=df[y], yerr=df[yerr],
             color="k", ms=3.5, fmt="o", label=label, expected_xs=expected_xs, add_ends=False)
        return
    if dataset_order is not None:
        input_datasets = df.index.unique(dataset_col)
        dataset_order = dataset_order + [d for d in input_datasets if d not in dataset_order]
    n_datasets = df.groupby(level=dataset_col).count()
    n_datasets = len(n_datasets[n_datasets != 0])

    vals = df[y].unstack(dataset_col).fillna(method="ffill", axis="columns")
    if kind == "line":
        filler = FillColl(n_datasets, ax=ax, fill=False, colourmap=colourmap,
                          dataset_colours=dataset_colours,
                          dataset_order=dataset_order, expected_xs=expected_xs)
        vals.apply(filler, axis=0, step="mid")
        return
    elif kind == "bar":
        filler = BarColl(n_datasets, ax=ax, colourmap=colourmap,
                         dataset_order=dataset_order, expected_xs=expected_xs)
        vals.apply(filler, axis=0, step="mid")
    elif kind == "fill":
        filler = FillColl(n_datasets, ax=ax, colourmap=colourmap,
                          dataset_colours=dataset_colours,
                          dataset_order=dataset_order,
                          line=False, expected_xs=expected_xs)
        vals.iloc[:, ::-1].apply(filler, axis=0, step="mid")
    elif kind == "fill-error-last":
        actually_plot(df, x_axis, y, yerr, "fill", label, ax, dataset_colours=dataset_colours,
                      dataset_col=dataset_col, colourmap=colourmap, dataset_order=dataset_order)
        summed = df.unstack(dataset_col).fillna(method="ffill", axis="columns")
        last_dataset = summed.columns.get_level_values(1)[n_datasets - 1]
        summed = summed.xs(last_dataset, level=1, axis="columns")
        x = summed.index.values
        y_down = (summed[y] - summed[yerr]).values
        y_up = (summed[y] + summed[yerr]).values
        draw(ax, "fill_between", x, ys=["y1", "y2"], y2=y_down, y1=y_up,
             color="gray", step="mid", alpha=0.7, expected_xs=expected_xs)
    else:
        raise RuntimeError("Unknown value for 'kind', '{}'".format(kind))


def standardize_values(x, y_values=[], fill_val=0, expected_xs=None, add_ends=True):
    """
    Standardize a set of arrays so they're ready to be plotted directly for matplotlib

    Algorithm:
    if any requested X values are missing:
        insert dummy values into X and Y values at the right location
    """
    if expected_xs is not None:
        x, y_values = add_missing_vals(x, expected_xs, y_values=y_values, fill_val=fill_val)

    if x.dtype.kind in 'bifc':
        x = replace_infs(x)

        if add_ends:
            x, y_values = pad_ends(x, y_values=y_values, fill_val=fill_val)
    return (x,) + tuple(y_values)


def replace_infs(x):
    """
    Replace (pos or neg) infinities at the ends of an array of floats

    Algorithm: X has +/- inf at an end, replace this X value with +/- the
    previous/next value of X +/- the mean width in X
    """
    x = x[:]  # Make a copy of the array
    is_left_inf = np.isneginf(x[0])
    is_right_inf = np.isposinf(x[-1])
    width_slice = x[1 if is_left_inf else None:-1 if is_right_inf else None]
    mean_width = width_slice[0]
    if len(width_slice) > 1:
        mean_width = np.diff(width_slice).mean()
    if is_left_inf:
        x[0] = x[1] - mean_width
    if is_right_inf:
        x[-1] = x[-2] + mean_width
    return x


def add_missing_vals(x, expected_xs, y_values=[], fill_val=0):
    """
    Check from a list of expected x values, if all occur in x.  If any are missing
    """
    insert = np.isin(expected_xs, x)
    new_ys = []
    for y in y_values:
        new = np.full_like(expected_xs, fill_val, dtype=y.dtype)
        new[insert] = y
        new_ys.append(new)
    if isinstance(expected_xs, (pd.Index, pd.MultiIndex)):
        new_x = expected_xs.values
    else:
        new_x = expected_xs.copy()
    return new_x, new_ys


def pad_ends(x, y_values=[], fill_val=0):
    """
    Insert a dummy entry to X and Y for all arrays
    """
    mean_width = x[0]
    if len(x) > 1:
        mean_width = np.diff(x).mean()

    x = np.concatenate((x[0:1] - mean_width, x, x[-1:] + mean_width), axis=0)
    new_values = [np.concatenate(([fill_val], y, [fill_val]), axis=0) for y in y_values]
    return x, tuple(new_values)


def plot_1d_many(df, prefix="", data="data", signal=None, dataset_col="dataset",
                 plot_sims="stack", plot_data="sum", plot_signal=None,
                 kind_data="scatter", kind_sims="fill-error-last", kind_signal="line",
                 scale_sims=None, summary="ratio-error-both", colourmap="nipy_spectral",
                 dataset_order=None, figsize=(5, 6), show_over_underflow=False,
                 dataset_colours=None, err_from_sumw2=False, **kwargs):
    y = "sumw"
    yvar = "sumw2"
    yerr = "err"
    if prefix:
        y = prefix + ":" + y
        yvar = prefix + ":" + yvar
        yerr = prefix + ":" + yerr

    df = utils.convert_intervals(df, to="mid")
    if not show_over_underflow:
        df = utils.drop_over_underflow(df)
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
        fig, main_ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": (3, 1)}, sharex=True, figsize=figsize)
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
        merged = _merge_datasets(df, combine, dataset_col, param_name=var_name, err_from_sumw2=err_from_sumw2)
        actually_plot(merged, x_axis=x_axis, y=y, yerr=yerr, kind=style,
                      label=label, ax=main_ax, dataset_col=dataset_col,
                      dataset_colours=dataset_colours,
                      colourmap=colourmap, dataset_order=dataset_order)
    main_ax.set_xlabel(x_axis)

    if not summary:
        return main_ax, None

    summary_ax = ax[1]
    err_msg = "Unknown value for summary, '{}'".format(summary)
    if summary.startswith("ratio"):
        main_ax.set_xlabel("")
        summed_data = _merge_datasets(
            in_df_data, "sum", dataset_col=dataset_col, err_from_sumw2=err_from_sumw2)
        summed_sims = _merge_datasets(
            in_df_sims, "sum", dataset_col=dataset_col, err_from_sumw2=err_from_sumw2)
        if summary == "ratio-error-both":
            error = "both"
        elif summary == "ratio-error-markers":
            error = "markers"
        else:
            raise RuntimeError(err_msg)
        if 'ratio_ylim' not in kwargs.keys():
            kwargs['ratio_ylim'] = [0., 2.]
        plot_ratio(summed_data, summed_sims, x=x_axis,
                   y=y, yerr=yerr, ax=summary_ax, error=error, ylim=kwargs['ratio_ylim'])
    else:
        raise RuntimeError(err_msg)
    return main_ax, summary_ax


def _merge_datasets(df, style, dataset_col, param_name="_merge_datasets", err_from_sumw2=False):
    if style == "stack":
        utils.calculate_error(df, do_rel_err=not err_from_sumw2)
        df = utils.stack_datasets(df, dataset_level=dataset_col)
    elif style == "sum":
        df = utils.sum_over_datasets(df, dataset_level=dataset_col)
        utils.calculate_error(df, do_rel_err=not err_from_sumw2)
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


def plot_ratio(data, sims, x, y, yerr, ax, error="both", ylim=[0., 2]):
    # make sure both sides agree with the binning
    merged = data.join(sims, how="left", lsuffix="data", rsuffix="sims")
    data = merged.filter(like="data", axis="columns").fillna(0)
    data.columns = [col.replace("data", "") for col in data.columns]
    sims = merged.filter(like="sims", axis="columns")
    sims.columns = [col.replace("sims", "") for col in sims.columns]

    s, s_err = sims[y], sims[yerr]
    d, d_err = data[y], data[yerr]
    x_axis = data.reset_index()[x]

    if error == "markers":
        central, lower, upper = stats.try_root_ratio_plot(d, d_err, s, s_err)
        x_axis, central, lower, upper = standardize_values(x_axis, y_values=(central, lower, upper), add_ends=False)
        mask = (central != 0) & (lower != 0)
        ax.errorbar(x=x_axis[mask], y=central[mask], yerr=(lower[mask], upper[mask]),
                    fmt="o", markersize=4, color="k")

    elif error == "both":
        ratio = d / s
        rel_d_err = (d_err / s)
        rel_s_err = (s_err / s)

        vals = standardize_values(x_axis.values, y_values=[ratio, rel_s_err, rel_d_err], add_ends=False)
        x_axis, ratio, rel_s_err, rel_d_err = vals

        ax.errorbar(x=x_axis, y=ratio, yerr=rel_d_err, fmt="o", markersize=4, color="k")
        draw(ax, "fill_between", x_axis, ys=["y1", "y2"],
             y2=1 + rel_s_err, y1=1 - rel_s_err, fill_val=1,
             color="gray", step="mid", alpha=0.7)

    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(x)
    ax.set_ylabel("Data / MC")


def draw(ax, method, x, ys, **kwargs):
    fill_val = kwargs.pop("fill_val", 0)
    expected_xs = kwargs.pop("expected_xs", None)
    add_ends = kwargs.pop("add_ends", True)
    if x.dtype.kind in 'biufc':
        values = standardize_values(x, [kwargs[y] for y in ys],
                                    fill_val=fill_val,
                                    add_ends=add_ends,
                                    expected_xs=expected_xs)
        x = values[0]
        new_ys = values[1:]
        kwargs.update(dict(zip(ys, new_ys)))
        ticks = None
    else:
        x, ticks = np.arange(len(x)), x
    getattr(ax, method)(x=x, **kwargs)
    if ticks is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
    return x, ticks
