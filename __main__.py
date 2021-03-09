"""
Turn them tables into plots
"""
import os
import six
import logging
import matplotlib
import math
import numbers
import numpy as np
matplotlib.use('Agg')
matplotlib.rcParams.update({'figure.autolayout': True})
from .version import __version__ # noqa
from .utils import read_binned_df, weighting_vars, binning_vars # noqa
from .utils import decipher_filename, mask_rows  # noqa
from .plotting import plot_all, add_annotations, is_intervals # noqa


logger = logging.getLogger("fast_plotter")
logger.setLevel(logging.INFO)
HANDLER = logging.StreamHandler()
FORMATTER = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
HANDLER.setFormatter(FORMATTER)
logger.addHandler(HANDLER)


def arg_parser(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("tables", type=str, nargs="+",
                        help="Table files to process")
    parser.add_argument("-c", "--config", type=str,
                        help="YAML config to control common plotting options")
    parser.add_argument("-o", "--outdir", type=str, default="plots",
                        help="Output directory to save plots to")
    parser.add_argument("-e", "--extension", type=lambda x: x.split(","), default=["png"],
                        help="File extension for images")
    parser.add_argument("-w", "--weights", default=[], type=lambda x: x.split(","),
                        help="comma-separated list of weight schemes to plot things for")
    parser.add_argument("-d", "--data", default="data",
                        help="Regular expression to identify real data datasets from their name")
    parser.add_argument("-s", "--signal", default="signal",
                        help="Regular expression to identify signal MC datasets from their name")
    parser.add_argument("--dataset-col", default="dataset",
                        help="Name of column to be used to define multiple-lines for 1D plots")
    parser.add_argument("-l", "--lumi", default=None, type=float,
                        help="Scale the MC yields by this lumi")
    parser.add_argument("-y", "--yscale", default="log", choices=["log", "linear"],
                        help="Use this scale for the y-axis")
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

    def split_equals(arg):
        return arg.split("=")
    parser.add_argument("-v", "--variable", dest="variables", action="append", default=[], type=split_equals,
                        help="Define a variable to expand in the config file")
    parser.add_argument("--halt-errors", dest="continue_errors", default=True, action="store_false",
                        help="Stop at the first time an error occurs")
    return parser


def main(args=None):
    args = arg_parser().parse_args(args=args)
    config = getattr(args, "config", None)
    if config:
        args = process_cfg(config, args)

    if not os.path.exists(args.outdir):
        logger.info("Creating output directory " + args.outdir)
        os.makedirs(args.outdir)

    ran_ok = True
    for infile in args.tables:
        ran_ok &= process_one_file(infile, args)
    return 0 if ran_ok else 1


def process_cfg(cfg_file, args, make_arg_parser=None):
    if not make_arg_parser:
        make_arg_parser = arg_parser
    import yaml
    from argparse import Namespace
    from string import Template
    with open(cfg_file, "r") as infile:
        cfg = yaml.safe_load(infile)
    # Only way to neatly allow cmd-line args to override config and handle
    # defaults seems to be:
    parser = make_arg_parser()
    parser.set_defaults(**cfg)
    args = parser.parse_args()
    if args.variables:

        def recursive_replace(value, replacements):
            if isinstance(value, (tuple, list)):
                return type(value)([recursive_replace(v, replacements) for v in value])
            if isinstance(value, dict):
                return {k: recursive_replace(v, replacements) for k, v in value.items()}
            if isinstance(value, six.string_types):
                return Template(value).safe_substitute(replacements)
            return value

        replacements = dict(args.variables)
        args = Namespace(**recursive_replace(vars(args), replacements))

    return args

def autoscale_values(args, df_filtered, weight, data_rows, mc_rows, ylim_lower=0.1, legend_size=2):
    dims = tuple(dim for dim in binning_vars(df_filtered) if dim != args.dataset_col)
    limits={}
    if hasattr(args, "autoscale"):
        for dim in dims:
            if 'y' in args.autoscale:
                if weight == "n":
                    max_y = df_filtered['sumw'].max()
                else:
                    max_mc = df_filtered.loc[mc_rows, 'sumw'].max()*args.lumi
                    max_data = df_filtered.loc[data_rows, 'n'].max() if 'n' in df_filtered.columns else 0.1
                    max_y = max(max_mc, max_data)
                max_y = max_y if max_y >= 1 else 1
                if args.yscale=='log': 
                    ylim_upper_floor = math.floor(math.log10(max_y))
                    y_buffer = legend_size + 2 if ylim_upper_floor > 4 /
                               else legend_size if ylim_upper_floor > 2
                               else legend_size # Buffer to make room for legend
                    ylim_upper = float('1e'+str(ylim_upper_floor+y_buffer))
                    ylim_lower = 1e-1
                else:
                    ylim_upper = round(max_y*1.5, -int(math.floor(math.log10(abs(max_y))))) #Buffer for legend
                    ylim_lower = 0
                ylim = [ylim_lower, ylim_upper]
            else:
                ylim = args.limits['y'] if 'y' in args.limits else None
            df_aboveMin = df_filtered.loc[df_filtered['sumw']>ylim_lower] 
            xcol = df_aboveMin.index.get_level_values(dim)
            if 'x' in args.autoscale: #Determine x-axis limits
                if is_intervals(xcol): #If x-axis is interval, take rightmost and leftmost intervals unless they are inf
                    max_x = xcol.right.max() if np.isfinite(xcol.right.max()) else xcol.left.max()
                    min_x = xcol.left.min() if np.isfinite(xcol.left.min()) else xcol.right.min()
                    if not (np.isfinite(max_x) and hasattr(args, "show_over_underflow") and args.show_over_underflow:
                        logger.warn("Cannot autoscale overflow bin for x-axis. Removing.")
                    xlim = [min_x, max_x]
                elif isinstance(xcol, numbers.Number):
                    xlim = [xcol.min, xcol.max]
                else:
                    xlim = [-0.5, len(xcol.unique())-0.5] #For non-numeric x-axis (e.g. mtn range) 
            else:
                xlim = args.limits['x'] if 'x' in args.limits else None

            limits[dim]={"x":xlim, "y":ylim}
    else:
        xlim = args.limits['x'] if 'x' in args.limits else None
        ylim = args.limits['y'] if 'y' in args.limits else None
        limits  = {dim:{"x":xlim, "y":ylim} for dim in dims}
    return limits
        
     

def process_one_file(infile, args):
    df = read_binned_df(infile, dtype={args.dataset_col: str})
    weights = weighting_vars(df)
    autoscale = hasattr(args, "autoscale")
    legend_size = args.legend_size if hasattr(args, "legend_size") else 2
    ran_ok = True
    for weight in weights:
        if args.weights and weight not in args.weights:
            continue
        df_filtered = df.copy()
        if weight == "n":
            df_filtered["sumw"] = df_filtered.n
            df_filtered["sumw2"] = df_filtered.n
            data_rows=None
            mc_rows=None
        else:
            data_rows = mask_rows(df_filtered,
                                  regex=args.data,
                                  level=args.dataset_col)
            mc_rows = mask_rows(df_filtered,
                                regex="^((?!"+args.data+").)*$",
                                level=args.dataset_col)
            if "n" in df.columns:
                for col in df_filtered.columns:
                    if col == "n":
                        continue
                    df_filtered.loc[data_rows, col] = df["n"][data_rows]
            df_filtered.columns = [
                n.replace(weight + ":", "") for n in df_filtered.columns]
        if hasattr(args, "value_replacements"):
            for column, replacements in args.value_replacements.items():
                if column not in df_filtered.index.names:
                    continue
                df_filtered.rename(replacements, level=column, inplace=True, axis="index")
                df_filtered = df_filtered.groupby(level=df.index.names).sum()
            data_rows = mask_rows(df_filtered,
                                  regex=args.data,
                                  level=args.dataset_col)
            mc_rows = mask_rows(df_filtered,
                                regex="^((?!"+args.data+").)*$",
                                level=args.dataset_col)
        args.limits = autoscale_values(args, df_filtered, weight, data_rows, mc_rows, legend_size)
        plots, ok = plot_all(df_filtered, **vars(args))
        ran_ok &= ok
        dress_main_plots(plots, **vars(args))
        save_plots(infile, weight, plots, args.outdir, args.extension)
    return ran_ok


def dress_main_plots(plots, annotations=[], yscale=None, ylabel=None, legend={},
                     limits={}, xtickrotation=None, autoscale=False, **kwargs):

    autoscale = True if not not autoscale else False
    for properties, (main_ax, summary_ax) in plots.items():
        projections = [prop[1] for prop in properties if prop[0]=="project"]
        x_axis = projections[0] if (len(projections)==1) else None
        dim_limits=limits[x_axis] if x_axis else list(limits.values())[0]
        add_annotations(annotations, main_ax, summary_ax)
        if yscale:
            main_ax.set_yscale(yscale)
        if ylabel:
            main_ax.set_ylabel(ylabel)
        main_ax.legend(**legend).set_zorder(20)
        main_ax.grid(True)
        main_ax.set_axisbelow(True)
        for axis, lims in dim_limits.items():
            if isinstance(lims, (tuple, list)):
                lims = map(float, lims)
                getattr(main_ax, "set_%slim" % axis)(*lims)
            elif lims.endswith("%"):
                 main_ax.margins(**{axis: float(lims[:-1])})
        if xtickrotation:
            matplotlib.pyplot.xticks(rotation=xtickrotation)


def save_plots(infile, weight, plots, outdir, extensions):
    binning, name = decipher_filename(infile)
    kernel = "plot_" + ".".join(binning)
    kernel += "--" + ".".join(name)
    kernel += "--" + weight
    kernel = os.path.join(outdir, kernel)
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]
    for properties, (main, ratio) in plots.items():
        insert = "-".join("%s_%s" % prop for prop in properties)
        path = kernel + "--" + insert
        for ext in extensions:
            path_ext = path + "." + ext
            logger.info("Saving plot: " + path_ext)
            plot = main.get_figure()
            plot.savefig(path_ext, dpi=200)
            matplotlib.pyplot.close(plot)


if __name__ == "__main__":
    main()
