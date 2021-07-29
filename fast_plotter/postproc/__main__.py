import logging
from . import stages, logger as base_logger
from .functions import open_many
from fast_flow.v1 import read_sequence_yaml
from fast_flow.help import argparse_help_stages
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-q', '--quiet', default="INFO",
                        action='store_const', const="WARNING", dest='verbosity',
                        help="quiet output (show errors and warnings only)")
    parser.add_argument("-d", "--debug-dfs",
                        action="store_const", const="DEBUG", dest='verbosity',
                        help="Print a dataframe after each step")
    parser.add_argument("--debug-dfs-query", default=None,
                        help="Provide a query to select rows from the debugged dataframe")
    parser.add_argument("--debug-rows", default=5, type=int,
                        help="Number of rows to dump from debugging dataframe")
    parser.add_argument("-p", "--post-process", default=None, required=True,
                        help="A yaml to configure the post-processing step")
    parser.add_argument("-o", "--outdir", default=".",
                        help="The name of the output directory")
    parser.add_argument("-V", "--value-cols", default=r"(.*sumw2?|^n$)",
                        help="A regular expression to control which columns are"
                        " identified as values and not bin labels")
    parser.add_argument("--help-stages", metavar="stage-name-regex", nargs="?", default=None,
                        action=argparse_help_stages(stages.known_stages,
                                                    "fast_plotter.postproc.stages",
                                                    full_output=False),
                        help="Print help specific to the available stages")
    parser.add_argument("--help-stages-full", metavar="stage",
                        action=argparse_help_stages(stages.known_stages,
                                                    "fast_plotter.postproc.stages",
                                                    full_output=True),
                        help="Print the full help specific to the available stages")
    parser.add_argument("files", nargs="+",
                        help="Input dataframes that need merging together")
    return parser


def read_processing_cfg(fname, out_dir):
    sequence = read_sequence_yaml(fname, backend=stages, output_dir=out_dir)
    return sequence


def debug_df(dfs, debug_dfs_query=""):
    if not debug_dfs_query:
        return dfs[0][0]

    for df in dfs:
        try:
            debug_df = df[0].query(debug_dfs_query)
            if not debug_df.empty:
                return debug_df
        except NameError:
            return None
    return None


def dump_debug_df(dfs, debug_dfs_query="", debug_rows=5):
    df = debug_df(dfs, debug_dfs_query)
    if df is None:
        logger.debug("No dataframes contain rows matching the debug-dfs-query")
    else:
        logger.debug(df.head(debug_rows).to_string())
    return df


def setup_logging(verbosity):
    level = getattr(logging, verbosity)
    base_logger.setLevel(level)
    return verbosity == "DEBUG"


def main(args=None):
    args = make_parser().parse_args(args=args)
    debug = setup_logging(args.verbosity)

    dfs = open_many(args.files, value_columns=args.value_cols)
    if debug:
        dump_debug_df(dfs, args.debug_dfs_query, args.debug_rows)

    sequence = read_processing_cfg(args.post_process, args.outdir)

    for stage in sequence:
        logger.info("Working on %d dataframes", len(dfs))
        dfs = stage(dfs)
        if debug:
            dump_debug_df(dfs, args.debug_dfs_query, args.debug_rows)


if __name__ == "__main__":
    main()
