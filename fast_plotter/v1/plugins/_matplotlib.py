from typing import Any
import matplotlib.pyplot as plt


DONE_SETUP = False


def setup_matplotlib():
    global DONE_SETUP
    if DONE_SETUP:
        return
    plt.rcParams['backend'] = 'Agg'
    DONE_SETUP = True


def set_limits(axis: plt.Axes, xlimits: tuple[int, int], ylimits: tuple[int, int]) -> None:
    setup_matplotlib()
    axis.set_xlim(xlimits)
    axis.set_ylim(ylimits)


def savefig(output_file_name: str, dpi: int = 300) -> None:
    setup_matplotlib()
    plt.rcParams['savefig.dpi'] = dpi  # this might become plugins.matplotlib.savefig_dpi
    plt.savefig(output_file_name)
    plt.close()


def subplots(*args: list[Any], **kwargs: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    setup_matplotlib()
    return plt.subplots(*args, **kwargs)


__all__ = [
    "plt",
    "savefig",
    "set_limits",
    "setup_matplotlib",
]
