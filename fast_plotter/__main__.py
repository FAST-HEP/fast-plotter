"""
Turn them tables into plots
"""
import os
import logging
import matplotlib
matplotlib.use('Agg')
from .utils import read_binned_df, weighting_vars, decipher_filename
from .plotting import plot_all, add_annotations


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
    parser.add_argument("-e", "--extension", type=str, default="png",
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
    return parser


def main(args=None):
    if args is None:
        args = arg_parser().parse_args(args=args)
    config = getattr(args, "config", None)
    if config:
        args = process_cfg(config, args)

    for infile in args.tables:
        process_one_file(infile, args)


def process_cfg(cfg_file, args):
    import yaml
    with open(cfg_file, "r") as infile:
        cfg = yaml.load(infile)
    # Only way to neatly allow cmd-line args to override config and handle
    # defaults seems to be:
    parser = arg_parser()
    parser.set_defaults(**cfg)
    args = parser.parse_args()

    return args


def process_one_file(infile, args):
    logger.info("Processing: " + infile)
    df = read_binned_df(infile, dtype={args.dataset_col: str})
    if hasattr(args, "value_replacements"):
        for column, replacements in args.value_replacements.items():
            df.rename(replacements, level=column, inplace=True, axis="index")
    weights = weighting_vars(df)
    for weight in weights:
        if args.weights and weight not in args.weights:
            continue
        if weight == "n":
            df_filtered = df.filter(weight, axis="columns").copy()
            df_filtered.rename({weight: "sumw"}, axis="columns", inplace=True)
            df_filtered["sumw2"] = df_filtered.sumw
        else:
            df_filtered = df.filter(like=weight, axis="columns").copy()
            if "n" in df.columns:
                isnull = df_filtered.isnull()
                for col in df_filtered.columns:
                    df_filtered[col][isnull[col]] = df["n"][isnull[col]]
            df_filtered.columns = [
                n.replace(weight + ":", "") for n in df_filtered.columns]
        plots = plot_all(df_filtered, infile + "__" + weight, **vars(args))
        dress_main_plots(plots, **vars(args))
        save_plots(infile, weight, plots, args.outdir, args.extension)


def dress_main_plots(plots, annotations=[], yscale=None, ylabel=None, legend={}, limits={}, **kwargs):
    for main_ax, _ in plots.values():
        add_annotations(annotations, main_ax)
        if yscale:
            main_ax.set_yscale(yscale)
        if ylabel:
            main_ax.set_ylabel(ylabel)
        main_ax.legend(**legend)
        main_ax.grid(True)
        main_ax.set_axisbelow(True)
        for axis, lims in limits.items():
            lims = map(float, lims)
            if axis.lower() == "x":
                main_ax.set_xlim(*lims)
            if axis.lower() == "y":
                main_ax.set_ylim(*lims)


def save_plots(infile, weight, plots, outdir, extension):
    binning, _ = decipher_filename(infile)
    kernel = "plot_" + ".".join(binning)
    kernel += "--" + weight
    kernel = os.path.join(outdir, kernel)
    for properties, (main, ratio) in plots.items():
        insert = "-".join("%s_%s" % prop for prop in properties)
        path = kernel + "--" + insert
        path += "." + extension
        logger.info("Saving plot: " + path)
        plot = main.get_figure()
        plot.savefig(path, dpi=200)
        matplotlib.pyplot.close(plot)


if __name__ == "__main__":
    args = arg_parser().parse_args()
    main(args)
