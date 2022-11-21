from dataclasses import dataclass, field
from omegaconf import OmegaConf
from typing import Any


@dataclass
class LabelConfig:
    x_label: str = "x"
    y_label: str = "y"
    title: str = "title"
    draw_title: bool = True


@dataclass
class CollectionConfig:
    style: str = "default"
    type: str = "efficiency"
    source_type: str = "root"
    labels: LabelConfig = field(default_factory=LabelConfig)

@dataclass
class StylesConfig:
    draw_legend: bool
    draw_title: bool # sets LabelConfig.draw_title
    plugins: dict[str, Any] = field(default_factory=dict)


def apply_style_to_collection(collection: CollectionConfig, style: StylesConfig):
    collection.labels.draw_title = style.draw_title
    collection.plugins = style.plugins


@dataclass
class PlotConfig:
    plotconfig_version: str = "1"
    styles: dict[str, StylesConfig] = field(default_factory=dict)
    collections: dict[str, CollectionConfig] = field(default_factory=dict)


def create_example_config(output_file_name: str):
    conf = OmegaConf.structured(PlotConfig)
    conf.collections["test-2"] = CollectionConfig()
    conf.styles["default"] = StylesConfig(
        draw_legend=True,
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


if __name__ == "__main__":
    create_example_config("example_config.yaml")
