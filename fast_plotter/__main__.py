"""
Turn them tables into plots
"""
import os
import six
import logging
import matplotlib
matplotlib.use('Agg')
from .utils import read_binned_df, weighting_vars # noqa
from .utils import decipher_filename, mask_rows  # noqa
from .plotting import plot_all, add_annotations # noqa


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


def process_cfg(cfg_file, args):
    import yaml
    from argparse import Namespace
    from string import Template
    with open(cfg_file, "r") as infile:
        cfg = yaml.safe_load(infile)
    # Only way to neatly allow cmd-line args to override config and handle
    # defaults seems to be:
    parser = arg_parser()
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


def process_one_file(infile, args):
    logger.info("Processing: " + infile)
    df = read_binned_df(infile, dtype={args.dataset_col: str})
    weights = weighting_vars(df)
    ran_ok = True
    for weight in weights:
        if args.weights and weight not in args.weights:
            continue
        df_filtered = df.copy()
        if weight == "n":
            df_filtered["sumw"] = df_filtered.n
            df_filtered["sumw2"] = df_filtered.n
        else:
            if "n" in df.columns:
                data_rows = mask_rows(df_filtered,
                                      regex=args.data,
                                      level=args.dataset_col)
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
        plots, ok = plot_all(df_filtered, **vars(args))
        ran_ok &= ok
        dress_main_plots(plots, **vars(args))
        save_plots(infile, weight, plots, args.outdir, args.extension)
    return ran_ok


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
            if isinstance(lims, (tuple, list)):
                lims = map(float, lims)
                if axis.lower() in "xy":
                    getattr(main_ax, "set_%slim" % axis)(*lims)
            elif lims.endswith("%"):
                main_ax.margins(**{axis: float(lims[:-1])})


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
