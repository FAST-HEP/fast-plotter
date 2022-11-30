from .plugins._mplhep import set_experiment_style, draw_experiment_label, histplot
from .plugins._matplotlib import savefig, setup_matplotlib, subplots, plt
from .plot import draw_legend, set_lables, draw_grid_overlay, modify_axes
from dataclasses import field
import os
import numpy as np
from typing import Any

from hist.intervals import ratio_uncertainty


class EfficiencyHistCollection():
    hists: list[Any] = field(default_factory=list)
    hist_colors: list[str] = field(default_factory=list)
    histtype = "errorbar"

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.hists = []
        self.hist_colors = []

    def add_hist(self, name, numerator, denominator, **kwargs):
        self.hists.append(Efficiency(name, numerator, denominator, **kwargs))

    def plot(self, **kwargs):
        """Plot the histograms in the collection.
        Compiles collection specific settings and passes them to the plotter.
        """
        setup_matplotlib()
        fig, ax = subplots()

        experiment = self.config.plugins["mplhep"]["experiment"]
        set_experiment_style(experiment)
        if "label_kwargs" in self.config.plugins["mplhep"]:
            draw_experiment_label(experiment, **self.config.plugins["mplhep"]["label_kwargs"])
        else:
            draw_experiment_label(experiment, fontsize=14, data=False, rlabel="13 TeV")

        plots = [hist.eff for hist in self.hists]
        labels = [hist.name for hist in self.hists]
        yerrs = [hist.eff_err for hist in self.hists]
        histplot(
            plots,
            markersize=8,
            stack=False,
            yerr=yerrs,
            xerr=True,
            label=labels,
            histtype=self.histtype,
            capsize=2,
            color=self.hist_colors,
            **kwargs,
        )
        draw_legend(ax, self.config.legend)
        set_lables(ax, self.config.labels)
        modify_axes(ax, self.config.axes)

        draw_grid_overlay(ax, self.config.grid_overlay)
        # plt.tight_layout()

    def save(self, output_dir):
        output_file = os.path.join(output_dir, f"{self.name}.png")
        print(f"Saving {output_file}")
        savefig(output_file)


class Efficiency:
    num: np.ndarray
    den: np.ndarray
    name: str

    def __init__(self, name, num, den):
        self.name = name
        self.num = num
        self.den = den
        self._eff = None
        self._eff_err = None

    @property
    def eff(self):
        if self._eff is None:
            old_settings = np.seterr()
            np.seterr(divide='ignore', invalid='ignore')
            self._eff = np.divide(self.num, self.den, dtype=np.float64)
            np.seterr(**old_settings)
            self._eff[np.isnan(self._eff)] = 0.0

        return self._eff

    @property
    def eff_err(self):
        if self._eff_err is None:
            if np.any(self.num > self.den):
                raise ValueError(
                    "Found numerator larger than denominator while calculating binomial uncertainty"
                )
            self._eff_err = ratio_uncertainty(self.num, self.den, uncertainty_type="efficiency")
        return self._eff_err

    def plot(self, **kwargs):
        histplot(self.eff, yerr=self.eff_err, **kwargs)

    def __repr__(self):
        return f"EfficiencyHist(num={self.num}, den={self.den})"
