from dataclasses import dataclass, field
from typing import Any

# TODO: tie together with OmegaConf

LineStyle = str | tuple[int, tuple[int, int]]
@dataclass
class CanvasSettings:
    width: int = 1920
    height: int = 1080
    dpi: int = 300
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class LabelSettings:
    x_label: str
    y_label: str
    title: str

    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LegendSettings:
    show_legend: bool = True
    legend_loc: str = "best"
    legend_ncol: int = 1
    legend_fontsize: int = 12

    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class GridSettings:
    show_grid: bool = True
    linestyle: LineStyle = (0, (5, 5))
    linewidth: float = 1
    alpha: float = 0.5
    color: str = "grey"
    vertical_lines: list[float] = field(default_factory=list)
    horizontal_lines: list[float] = field(default_factory=list)
    ylimits: tuple[float, float] = (0, 1)
    xlimits: tuple[float, float] = (0, 1)

    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class TickSettings:
    show_ticks: bool = True
    tick_direction: str = "in"
    tick_length: int = 5
    tick_width: int = 1
    tick_labelsize: int = 12
    kwargs: dict[str, Any] = field(default_factory=dict)
