from matplotlib import container
import numpy as np

from .config import AxesConfig, LegendConfig, LabelConfig, GridOverlayConfig
from .plugins._matplotlib import plt

CUSTOM_LINE_STYLES = {
    "dashed-5-5": (0, (5, 5)),
}


def _remove_errorbars(handles):
    """Remove errorbars from a list of handles."""
    return [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]


def _replace_custom_linestyle(line_style):
    """Replace custom line styles with matplotlib line styles."""
    if line_style in CUSTOM_LINE_STYLES:
        return CUSTOM_LINE_STYLES[line_style]
    return line_style


def _auto_limits(current_limits: tuple[float, float], limits: tuple[float, float], increase: int = 0.1) -> tuple[float, float]:
    """Automatically determine the limits for the given axis."""
    if limits is None:
        return current_limits
    if limits[0] is None or limits[0] == -np.inf:
        factor = 1 + increase if current_limits[0] < 0 else 1 - increase
        limits = (current_limits[0] * factor, limits[1])
    if limits[1] is None or limits[1] == np.inf:
        factor = 1 + increase if current_limits[1] > 0 else 1 - increase
        limits = (limits[0], current_limits[1] * factor)
    return limits


def draw_legend(axis: plt.Axes, settings: LegendConfig) -> None:
    """Draw a legend on the given axis. Content must be already drawn when calling this function."""
    if not settings.show_legend:
        return
    handles, labels = axis.get_legend_handles_labels()
    handles = _remove_errorbars(handles)
    axis.legend(handles, labels, title="$|\eta| < 2.4$", **settings.kwargs)


def set_lables(axis: plt.Axes, labels: LabelConfig) -> None:
    axis.set_xlabel(labels.xlabel)
    axis.set_ylabel(labels.ylabel)
    if labels.show_title:
        axis.set_title(labels.title)


def draw_grid_overlay(axis: plt.Axes, grid: GridOverlayConfig) -> None:
    # check if linestyle is OK
    line_style = _replace_custom_linestyle(grid.linestyle)
    xlimits = _auto_limits(axis.get_xlim(), grid.xlimits)
    ylimits = _auto_limits(axis.get_ylim(), grid.ylimits)
    axis.vlines(
        grid.vertical_lines,
        color=grid.color,
        linestyle=line_style,
        linewidth=grid.linewidth,
        ymin=ylimits[0],
        ymax=ylimits[1],
        alpha=grid.alpha,
        **grid.kwargs,
    )
    axis.hlines(
        grid.horizontal_lines,
        color=grid.color,
        linestyle=line_style,
        xmin=xlimits[0],
        xmax=xlimits[1],
        linewidth=grid.linewidth,
        alpha=grid.alpha,
        **grid.kwargs,
    )


def modify_axes(axis: plt.Axes, config: AxesConfig) -> None:
    """Sets the axes limits, scale (e.g. log), and ticks."""
    axis.set_xlim(config.xlimits)
    axis.set_ylim(config.ylimits)
    axis.set_xscale("log" if config.xlog else "linear")
    axis.set_yscale("log" if config.ylog else "linear")
    axis.xaxis.set_major_locator(plt.MaxNLocator(config.major_ticks))
    axis.xaxis.set_minor_locator(plt.MaxNLocator(config.minor_ticks))
    axis.yaxis.set_major_locator(plt.MaxNLocator(config.major_ticks))
    axis.yaxis.set_minor_locator(plt.MaxNLocator(config.minor_ticks))
    axis.tick_params(which="major", length=10, width=1, direction="in")
    axis.tick_params(which="minor", length=5, width=1, direction="in")

# def set_ticks(axis: plt.Axes, ticks: TickSettings) -> None:
#     pass

# def plot_efficiency(collection: EfficiencyCollection, settings: PlotSettings) -> None:
#     pass
