from dataclasses import dataclass, field, fields
import numpy as np
from omegaconf import OmegaConf
from typing import Any

from fasthep_logging import get_logger

logger = get_logger()


@dataclass
class LabelConfig:
    xlabel: str = field(default="x")
    ylabel: str = field(default="y")
    title: str = field(default="title")
    show_title: bool = field(default=True)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LegendConfig:
    show: bool = field(default=True)
    title: str = field(default="title")
    show_title: bool = field(default=False)
    legend_loc: str = field(default="best")
    legend_ncol: int = field(default=1)
    legend_fontsize: int = field(default=12)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GridOverlayConfig():
    show: bool = field(default=True)
    linestyle: str = field(default="dashed-5-5")
    linewidth: float = field(default=1)
    alpha: float = field(default=0.5)
    color: str = field(default="grey")
    vertical_lines: list[float] = field(default_factory=list)
    horizontal_lines: list[float] = field(default_factory=list)
    ylimits: tuple[float, float] = field(default=(-np.inf, np.inf))
    xlimits: tuple[float, float] = field(default=(-np.inf, np.inf))
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanvasConfig:
    width: int = field(default=1920)
    height: int = field(default=1080)
    dpi: int = field(default=300)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceConfig:
    path: str = field(default="")
    label: str = field(default="")
    color: str = field(default="black")
    protocol: str = field(default="root")


@dataclass
class AxesConfig:
    xlog: bool = field(default=False)
    ylog: bool = field(default=False)
    xlimits: tuple[float, float] = field(default=(-np.inf, np.inf))
    ylimits: tuple[float, float] = field(default=(-np.inf, np.inf))
    major_ticks: int = field(default=5)
    minor_ticks: int = field(default=5)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionConfig:
    style: str = field(default="default")
    type: str = field(default="efficiency")
    labels: LabelConfig = field(default_factory=LabelConfig)
    legend: LegendConfig = field(default_factory=LegendConfig)
    grid_overlay: GridOverlayConfig = field(default_factory=GridOverlayConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    axes: AxesConfig = field(default_factory=AxesConfig)
    plugins: dict[str, Any] = field(default_factory=dict)


@dataclass
class StylesConfig:
    """ StylesConfig sets defaults for CollectionConfig.
    A style can be used by more than one CollectionConfig.
    """
    legend: LegendConfig = field(default_factory=LegendConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    grid_overlay: GridOverlayConfig = field(default_factory=GridOverlayConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    axes: AxesConfig = field(default_factory=AxesConfig)
    plugins: dict[str, Any] = field(default_factory=dict)


def apply_style_to_collection(collection: CollectionConfig, style: StylesConfig) -> None:
    """Applies the style to the collection.
        Order of precedence:
        1. collection
        2. style
        3. default style
    """
    default_config = OmegaConf.structured(StylesConfig)
    supported_attributes = [field.name for field in fields(StylesConfig)]
    temp = None
    for attribute in supported_attributes:
        logger.debug(f"Applying {attribute} style to collection {collection}")
        in_collection = hasattr(collection, attribute)
        in_style = hasattr(style, attribute)

        if not in_collection and not in_style:
            setattr(collection, attribute, getattr(default_config, attribute))
            continue

        to_merge = [getattr(default_config, attribute)]
        if in_collection:
            to_merge.append(getattr(collection, attribute))
        if in_style:
            to_merge.append(getattr(style, attribute))
        temp = OmegaConf.merge(*to_merge)

        setattr(collection, attribute, temp)


@dataclass
class PlotConfig:
    plotconfig_version: str = field(default="1")
    styles: dict[str, StylesConfig] = field(default_factory=dict)
    collections: dict[str, CollectionConfig] = field(default_factory=dict)


def load_config(config_file_name: str) -> PlotConfig:
    logger.info(f"Loading config from {config_file_name}")
    plot_config = OmegaConf.load(config_file_name)
    # merge styles into collections
    for name, collection in plot_config.collections.items():
        style = plot_config.styles[collection.style]
        apply_style_to_collection(collection, style)

    return plot_config


def create_example_config(output_file_name: str):
    conf = OmegaConf.structured(PlotConfig)
    conf.collections["test-2"] = CollectionConfig()
    conf.styles["default"] = StylesConfig(
        plugins={
            "matplotlib": {
                "markersize": 6,
                "xerr": True,
            },
            "mplhep": {
                "experiment": "CMS",
            },
        },
    )
    with open(output_file_name, "w") as file_handle:
        OmegaConf.save(conf, file_handle)

    conf2 = load_config(output_file_name)


if __name__ == "__main__":
    create_example_config("docs/example_v1_config.yaml")
