import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import field
import os
import numpy as np
from typing import Any

from hist.intervals import ratio_uncertainty
import mplhep as hep
import matplotlib
matplotlib.use("Agg")

from .plot import draw_legend, set_lables, set_grid, set_ticks
from .settings import LabelSettings, LegendSettings, GridSettings, TickSettings
from .plugins._matplotlib import savefig, setup_matplotlib

class EfficiencyHistCollection():
    hists: list[Any] = field(default_factory=list)
    hist_colors: list[str] = field(default_factory=list)
    histtype = "errorbar"

    def __init__(self, name, title, style):
        self.name = name
        self.title = title
        self.style = style
        self.hists = []
        self.hist_colors = []

    def add_hist(self, name, numerator, denominator, **kwargs):
        self.hists.append(Efficiency(name, numerator, denominator, **kwargs))

    def plot(self, **kwargs):
        setup_matplotlib()
        fig, ax = plt.subplots()
        hep.style.use("CMS")
        hep.cms.label(data=False, fontsize=14)

        plots = [hist.eff for hist in self.hists]
        labels = [hist.name for hist in self.hists]
        yerrs = [hist.eff_err for hist in self.hists]
        hep.histplot(
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
        legend_settings = LegendSettings()
        draw_legend(ax, legend_settings)
        label_settings = LabelSettings(
            x_label="p_{T} [GeV]",
            y_label="Efficiency",
            title=self.title,
        )
        set_lables(ax, label_settings)
        # plt.xlabel("$p_{T}$ [GeV]")
        # plt.ylabel("Efficiency")
        # ax.tick_params(which='major', length=10, width=1, direction='in')
        # ax.tick_params(which='minor', length=5, width=1, direction='in')
        #xmin, xmax = ax.get_xlim()
        xmin, xmax = 0, 40
        ymin, ymax = 0, 1.1

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        grid_settings = GridSettings(
            vertical_lines=[10, 20, 30, 40],
            horizontal_lines=[0.25, 0.5, 0.75, 0.95, 1],
            xlimits=(xmin, xmax),
            ylimits=(ymin, 1),
        )
        set_grid(ax, grid_settings)
        plt.tight_layout()


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
        hep.histplot(self.eff, yerr=self.eff_err, **kwargs)

    def __repr__(self):
        return f"EfficiencyHist(num={self.num}, den={self.den})"
