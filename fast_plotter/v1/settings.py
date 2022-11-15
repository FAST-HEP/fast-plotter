from dataclass import dataclass
from typing import Any

# TODO: tie together with OmegaConf


@dataclass
class CanvasSettings:
    width: int
    height: int
    dpi: int


class LabelSettings:
    x_label: str
    y_label: str
    title: str
    kwargs: dict[str, Any]


@dataclass
class LegendSettings:
    show_legend: bool = True
    legend_loc: str = "best"
    legend_ncol: int = 1
    legend_fontsize: int = 12
