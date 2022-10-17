import os
import numpy as np

from hist.intervals import ratio_uncertainty
import mplhep as hep
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class EfficiencyHistCollection():

    def __init__(self, name, title, style):
        self.name = name
        self.title = title
        self.style = style
        self.hists = []

    def add_hist(self, name, numerator, denominator, **kwargs):
        self.hists.append(Efficiency(name, numerator, denominator, **kwargs))

    def plot(self, **kwargs):
        hep.style.use("CMS")
        fig = plt.figure(figsize=(10, 8))
        for hist in self.hists:
            hep.histplot(hist.eff, yerr=hist.eff_err, **kwargs)
    
    def save(self, output_dir):
        output_file = os.path.join(output_dir, f"{self.name}.png")
        print(f"Saving {output_file}")
        plt.savefig(output_file)


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
