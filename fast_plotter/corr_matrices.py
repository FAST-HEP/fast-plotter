# macro to produce two dataframes to be plotted in a correlation matrix

import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import fast_plotter.interval_from_str as ifs
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
import seaborn as sns

# List of dataframe .csv filenames
dffilename = ["tbl_dataset.region.dphi_met_lbj.dphi_met_lj.category--cut_lbj_vs_lj.csv", "tbl_dataset.region.dphi_met_lbj.dphi_met_sbj.category--lbj_vs_sbj.csv"]

# y-axis variable, the first variable column in the dataframe
df_col1 = ["dphi_met_lbj", "dphi_met_lbj"]
# x-axis variable, the second variable column in the dataframe
df_col2 = ["dphi_met_lj", "dphi_met_sbj"]

# Process label, inclusive in each category
dfproc = ["_ttH", "_VH", "_ggF"]

# Latex-formatted y-axis label
latex_1 = [r"$\Delta \phi(p_{T}^{miss},j_{b1})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{b1})$ (rad)"]

# example_latex_1 = [r"$p_{T}^{miss}$ (GeV)", r"$p_{T}^{miss}$ (GeV)", r"$\Delta \phi(p_{T}^{miss},j_{1} j_{2})$ (rad)",
#        r"$\Delta \phi(p_{T}^{miss},j_{b1} j_{b2})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{1} j_{2})$ (rad)",
#        r"$\Delta \phi(p_{T}^{miss},j_{b1})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{b1})$ (rad)",
#        r"$\Delta \phi(p_{T}^{miss},j_{1})$ (rad)", r"$p_{T}^{miss}$ (GeV)", r"$\Delta \phi(p_{T}^{miss},j_{b1})$ (rad)"]

# Latex-formatted x-axis label
latex_2 = [r"$\Delta \phi(p_{T}^{miss},j_{1})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{b2})$ (rad)"]

# example_latex_2 = [r"$\Delta \phi(j_{1},j_{2})$ (rad)", r"$\Delta \phi(j_{b1},j_{b2})$ (rad)", r"$\Delta \phi(j_{1},j_{2})$ (rad)",
#        r"$\Delta \phi(j_{b1},j_{b2})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{b1})$ (rad)",
#        r"$\Delta \phi(p_{T}^{miss},j_{1})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{b2})$ (rad)",
#        r"$\Delta \phi(p_{T}^{miss},j_{2})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{b1})$ (rad)", r"$\Delta \phi(p_{T}^{miss},j_{2})$ (rad)"]

# File containing the plots
os.mkdir('corr_plots')

for i in range(len(dffilename)):

        dftemp = pd.read_csv(dffilename[i])
        dftemp = dftemp[pd.notnull(dftemp['n'])]
        #dftemp = dftemp[dftemp['dataset'].str.contains("TTTo")] # to select a certain sample or set of samples in the dataframe

        ttH = dftemp.loc[dftemp['category'] > 1.005].loc[dftemp['category'] < 1.095]
        VH = dftemp.loc[dftemp['category'] > 2.005].loc[dftemp['category'] < 2.045]
        ggF = dftemp.loc[dftemp['category'] > 3.005].loc[dftemp['category'] < 3.045]
        dftemp.empty

        ttH.groupby(["dataset", df_col1[i], df_col2[i]]).sum().reset_index().drop(columns = ['category'])
        VH.groupby(["dataset", df_col1[i], df_col2[i]]).sum().reset_index().drop(columns = ['category'])
        ggF.groupby(["dataset", df_col1[i], df_col2[i]]).sum().reset_index().drop(columns = ['category'])

        list_of_dfs = [ttH, VH, ggF]
        for j in range(len(list_of_dfs)):	

                if list_of_dfs[j].empty == False:
                        var1 = ifs.interval_from_string(list_of_dfs[j][df_col1[i]])
                        var2 = ifs.interval_from_string(list_of_dfs[j][df_col2[i]])

                        print(list_of_dfs[j])
                        list_of_dfs[j][df_col1[i]] = var1.left
                        list_of_dfs[j][df_col2[i]] = var2.left

                        del list_of_dfs[j]['dataset']

                        list_of_dfs[j]['weight_nominal:sumw'] = list_of_dfs[j]['weight_nominal:sumw']*41530
                        dfmin = list_of_dfs[j]['weight_nominal:sumw'].min()
                        dfmax = list_of_dfs[j]['weight_nominal:sumw'].max()
                        #list_of_dfs[j]['weight_nominal:sumw'] = np.log10(list_of_dfs[j]['weight_nominal:sumw'])

                        list_of_dfs[j] = list_of_dfs[j].groupby([df_col1[i], df_col2[i]]).sum().reset_index()
                        list_of_dfs[j] = list_of_dfs[j].pivot(index = df_col1[i], columns = df_col2[i], values = 'weight_nominal:sumw')

                        plt.subplots(figsize=(8,8))
                        twodee = sns.heatmap(list_of_dfs[j], annot_kws={"size": 5}, cmap='coolwarm', xticklabels=True, yticklabels=True,
                                     cbar_kws={'ticks': [10e-3, 10e-2, 10e-1, 10e+0, 10e+1, 10e+2, 10e+3, 10e+4], 'label': 'Expected number of events'},
                                     norm=SymLogNorm(linthresh=0.01, vmin=dfmin, vmax=dfmax))

                        twodee.collections[0].colorbar.set_ticklabels(['10e-3', '10e-2', '10e-1', '10e+0', '10e+1', '10e+2', '10e+3', '10e+4'])

                        twodee.invert_yaxis()
                        ax1 = twodee.get_xticklabels()
                        ax2 = twodee.get_yticklabels()

                        plt.xticks(np.arange(len(ax1)), ax1, fontsize=8, rotation=90)
                        plt.yticks(np.arange(len(ax2)), ax2, fontsize=8, rotation=0)
                        plt.ylabel(latex_1[i])
                        plt.xlabel(latex_2[i])

                        list_of_dfs[j].empty
                        plt.savefig("corr_plots/plot_" + df_col1[i] + "_vs_" + df_col2[i] + dfproc[j] + ".pdf")
                        plt.savefig("corr_plots/plot_" + df_col1[i] + "_vs_" + df_col2[i] + dfproc[j] + ".png")
                        plt.clf()

        ttH.empty
        VH.empty
        ggF.empty
