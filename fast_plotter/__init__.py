# -*- coding: utf-8 -*-

"""Top-level package for fast-plotter."""

__author__ = """F.A.S.T"""
__email__ = 'fast-hep@cern.ch'
__version__ = '0.1.0'


from __future__ import print_function
import logging
logging.getLogger(__name__).setLevel(logging.INFO)



def keys(df, value_cols=["n", "nvar"]):
    return df.drop(columns=value_cols).columns.tolist()

def normalize(df, norms, group_col, value_cols=["n", "nvar"]):
    index_labels = df.index.names
    [norm_col] = norms.reset_index().drop(columns=group_col).columns.tolist()
    df = pd.merge(df.reset_index(), norms.reset_index(), on=group_col).set_index(index_labels)
    #df = df.merge(norms, on=group_col)
    #df = df.merge(norms)
    #df.join(norms, how='right')
    for value in value_cols:
        if "var" in value:
            df[value] = df[value] / (df[norm_col]*df[norm_col])
        else:
            df[value] = df[value] / df[norm_col]
    df.drop(columns=norm_col, inplace=True)
    return df

def norm(df, value_cols=["n", "nvar"], norm_var="n", groupby="process"):
    index_labels = df.index.names
    df_norm = df.groupby(groupby)[[norm_var]].sum().rename({norm_var: "norm"}, axis=1)
    df_norm = pd.merge(df.reset_index(), df_norm.reset_index(), on=[groupby]).set_index(index_labels)
    df_norm.n /= df_norm.norm
    df_norm.nvar /= df_norm.norm*df_norm.norm
    df_norm.drop("norm", axis=1, inplace=True)
    return df_norm

def rename_components(df):
    vals = df.component.str.extract("^(?:(.*?)(?:_(?:Tune|13TeV).*$)|(.*M125))")
    def cat_na(row):
        l = str(row[0])
        r = str(row[1])
        return l if l.lower() != "nan" else r
    vals = vals.apply(cat_na, axis=1)
    df.component = vals
    return df

def read(filename, directory, keys=None):
    file = os.path.join(d, filename)
    df = pd.read_table(file, delim_whitespace=True, comment="#")
    if not keys:
        df = pd.concat(dfs)
    else:
        if keys is True:
            keys = directories
        df = pd.concat(dfs, keys=keys).reset_index(level=0).rename({"level_0": "key"}, axis=1).reset_index(drop=True)

    return df

def read_concat(filename, directories, keys=None):
    files = [os.path.join(d, filename) for d in directories]
    dfs = [pd.read_table(f, delim_whitespace=True, comment="#") for f in files]
    if not keys:
        df = pd.concat(dfs)
    else:
        if keys is True:
            keys = directories
        df = pd.concat(dfs, keys=keys).reset_index(level=0).rename({"level_0": "key"}, axis=1).reset_index(drop=True)

    #rename_components(df)
    return df
    
def tidy(df, norms=None, value_cols=["n", "nvar"], group_col = "process"):
    if "luminosity" in df.columns:
        df.drop("luminosity", inplace=True, axis=1)
    
    if isinstance(norms, pd.DataFrame):
        df = normalize(df, norms, group_col)
    if value_cols:
        key_cols = keys(df, value_cols)
        key_cols.append(key_cols.pop(key_cols.index(group_col)))
        df.set_index(key_cols, inplace=True)
    if norms is True:
        df = norm(df, groupby=group_col)
    return df

def binned_plot_2d(df2d, x, y, values="n", sum_over=None, groupby="process", log=False,  **kwargs):
    from seaborn import heatmap
    kwargs.setdefault("cmap", "afmhot_r")

    keep_cols = [x, y, values]
    if groupby:
        keep_cols += groupby if isinstance(groupby, list) else [groupby]
    if sum_over:
        keep_cols += sum_over if isinstance(sum_over, list) else [sum_over]
        
    df = df2d[keep_cols]
    if sum_over:
        cols = [col for col in df.columns if col not in sum_over and col != values]
        df = df.groupby(cols)[[values]].sum().reset_index()
    
    if groupby:
        df_out = df.groupby(groupby).apply(lambda df: df.pivot(index=y, columns=x, values=values))
    else:
        df_out = df.pivot(index=y, columns=x, values=values)
 
    norm = None
    if log:
        vmax = df[values].max()
        vmin = df[df[values] > 0][values].min()
        print(vmax, vmin)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    if groupby:
        for name, df in df_out.groupby(groupby):
            plt.figure()
            df_tmp = df.reset_index(0, drop=True)
            heatmap(df_tmp, norm=norm, **kwargs)
            plt.title(name)
    else:            
        ax = heatmap(df_out,  norm=norm, **kwargs)

    return df_out
