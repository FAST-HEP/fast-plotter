from typing import Any

import mplhep

def inspect_plugin_config(plugin_config: dict[str, Any]) -> None:
    plugin_config["experiment"] = plugin_config.get("experiment", None)
    return plugin_config

def set_experiment_style(experiment: str) -> None:
    if not experiment:
        return
    mplhep.style.use(experiment)


def draw_experiment_label(experiment: str, **kwargs: dict[str, Any]) -> None:
    if not experiment:
        return
    exp = getattr(mplhep, experiment.lower())
    exp.label(**kwargs)

def histplot(*args: list[Any], **kwargs: dict[str, Any]) -> None:
    return mplhep.histplot(*args, **kwargs)