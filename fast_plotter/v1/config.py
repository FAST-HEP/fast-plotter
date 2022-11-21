from dataclasses import dataclass, field
from omegaconf import OmegaConf
from typing import Any, Optional
import json

from fasthep_logging import get_logger

logger = get_logger()


@dataclass
class LabelConfig:
    x_label: str = "x"
    y_label: str = "y"
    title: str = "title"
    show_title: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LegendConfig:
    show: bool = True
    legend_loc: str = "best"
    legend_ncol: int = 1
    legend_fontsize: int = 12
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GridOverlayConfig():
    show: bool = True
    linestyle: str = "dashed-5-5"
    linewidth: float = 1
    alpha: float = 0.5
    color: str = "grey"
    vertical_lines: list[float] = field(default_factory=list)
    horizontal_lines: list[float] = field(default_factory=list)
    ylimits: tuple[float, float] = (0, 1)
    xlimits: tuple[float, float] = (0, 1)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanvasConfig:
    width: int = 1920
    height: int = 1080
    dpi: int = 300
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceConfig:
    path: str = ""
    label: str = ""
    color: str = "black"


@dataclass
class CollectionConfig:
    style: str = "default"
    type: str = "efficiency"
    source_type: str = "root"
    labels: LabelConfig = field(default_factory=LabelConfig)
    legend: LegendConfig = field(default_factory=LegendConfig)
    grid_overlay: GridOverlayConfig = field(default_factory=GridOverlayConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    plugins: dict[str, Any] = field(default_factory=dict)


@dataclass
class StylesConfig:
    """ StylesConfig sets defaults for CollectionConfig.
    A style can be used by more than one CollectionConfig.
    """
    legend: LegendConfig = field(default_factory=LegendConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    grid_overlay: Optional[GridOverlayConfig] = field(default_factory=GridOverlayConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    plugins: dict[str, Any] = field(default_factory=dict)


def apply_style_to_collection(collection: CollectionConfig, style: StylesConfig) -> None:
    for attribute in ["legend", "labels", "grid_overlay", "canvas", "plugins"]:
        logger.info(f"Applying style {attribute} to collection {collection}")
        logger.info(f"Collection {attribute} before: {getattr(collection, attribute)}")

        temp = OmegaConf.merge(getattr(collection, attribute), getattr(style, attribute))
        logger.info(f"Collection {attribute} after: {temp}")
        setattr(collection, attribute, temp)


@dataclass
class PlotConfig:
    plotconfig_version: str = "1"
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
                "style": "CMS",
            },
        },
    )
    with open(output_file_name, "w") as file_handle:
        OmegaConf.save(conf, file_handle)

    conf2 = load_config(output_file_name)


if __name__ == "__main__":
    create_example_config("docs/example_config.yaml")
