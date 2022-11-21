import matplotlib.pyplot as plt
from matplotlib import container

from .settings import LegendSettings, LabelSettings, GridSettings, TickSettings

CUSTOM_LINE_STYLES = {
    "dashed-5-5": (0, (5, 5)),
}


def _remove_errorbars(handles):
    """Remove errorbars from a list of handles."""
    return [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

def draw_legend(axis: plt.Axes, settings: LegendSettings) -> None:
    """Draw a legend on the given axis. Content must be already drawn when calling this function."""
    if not settings.show_legend:
        return
    handles, labels = axis.get_legend_handles_labels()
    handles = _remove_errorbars(handles)
    axis.legend(handles, labels, title="$|\eta| < 2.4$", **settings.kwargs)

def set_lables(axis: plt.Axes, labels: LabelSettings) -> None:
    # hep.cms.label(data=False)
    axis.set_xlabel(labels.x_label)
    axis.set_ylabel(labels.y_label)
    # axis.set_title(labels.title)

def set_grid(axis: plt.Axes, grid: GridSettings) -> None:
    axis.vlines(
        grid.vertical_lines,
        color=grid.color,
        linestyle=grid.linestyle,
        linewidth=grid.linewidth,
        ymin=grid.ylimits[0],
        ymax=grid.ylimits[1],
        alpha=grid.alpha,
        **grid.kwargs,
    )
    axis.hlines(
        grid.horizontal_lines,
        color=grid.color,
        linestyle=grid.linestyle,
        xmin=grid.xlimits[0],
        xmax=grid.xlimits[1],
        linewidth=grid.linewidth,
        alpha=grid.alpha,
        **grid.kwargs,
    )

def set_ticks(axis: plt.Axes, ticks: TickSettings) -> None:
    pass

# def plot_efficiency(collection: EfficiencyCollection, settings: PlotSettings) -> None:
#     pass
