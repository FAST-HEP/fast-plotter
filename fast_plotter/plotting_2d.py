from . import utils as utils
import traceback
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
import seaborn as sns

#might want to eventually add a z scale option
#
def plot_all_2d(df, project_1d=True, project_2d=True, data="data", signal=None, dataset_col="dataset",
             yscale="log", lumi=None, annotations=[], dataset_order="sum-ascending",
             continue_errors=True, bin_variable_replacements={}, **kwargs):

    dimensions = utils.binning_vars(df)
    ran_ok = True
    
    if len(dimensions) == 2: #if there is no dataset specified in df
        df = utils.rename_index(df, bin_variable_replacements)
        figures[(("yscale", yscale),)] = plot_2d(
            df, yscale=yscale, annotations=annotations)

    if dataset_col in dimensions: # check how many dimensions there are excluding dataset 
        dimensions = tuple(dim for dim in dimensions if dim != dataset_col)

    if project_2d and len(dimensions) > 2: #if we want a 2d hist and the dimensions are >2
        # this will be more complicated once we add the ability to include multiple datasets 
        df = utils.rename_index(df, bin_variable_replacements)
        figures[(("yscale", yscale),)] = plot_2d(
            df, yscale=yscale, annotations=annotations)

        


def plot_2d_many():
    print("many")



def plot_2d(df):
    dimensions = utils.binning_vars(df) # tuple of column names 
    df.reset_index(inplace=True) # comvert from multiindex to columns
    reshaped = df.pivot(dimensions[2],dimensions[1],dimensions[3]) # reshape columns and rows 
    f,ax = plt.subplots() # can be generalised for multiple datasets
    sns.heatmap(reshaped,annot=True,ax=ax) # annot could take annotations array? 
    return f
