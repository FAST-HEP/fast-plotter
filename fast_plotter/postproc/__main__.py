import logging
from . import stages
from .functions import open_many
from fast_flow.v1 import read_sequence_yaml
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug-dfs", default=False, action="store_true",
                        help="Print a dataframe after each step")
    parser.add_argument("--debug-dfs-query", default=None,
                        help="Provide a query to select rows from the debugged dataframe")
    parser.add_argument("--debug-rows", default=5, type=int,
                        help="Number of rows to dump from debugging dataframe")
    parser.add_argument("-p", "--post-process", default=None, required=True,
                        help="A yaml to configure the post-processing step")
    parser.add_argument("-o", "--outdir", default=".",
                        help="The name of the output directory")
    parser.add_argument("files", nargs="+",
                        help="Input dataframes that need merging together")
    return parser


def read_processing_cfg(fname, out_dir):
    sequence = read_sequence_yaml(fname, backend=stages, output_dir=out_dir)
    return sequence


def dump_debug_df(dfs, debug_dfs_query=""):
    if not debug_dfs_query:
        return dfs[0][0]

    for df in dfs:
        try:
            debug_df = df[0].query(debug_dfs_query)
            if not debug_df.empty:
                return debug_df
        except NameError:
            return None

    logger.debug("No dataframes contain rows matching the debug-dfs-query")
    return None


def main(args=None):
    args = make_parser().parse_args(args=args)
    if args.debug_dfs:
        logger.setLevel(logging.DEBUG)

    dfs = open_many(args.files)
    sequence = read_processing_cfg(args.post_process, args.outdir)

    for stage in sequence:
        logger.info("Working on %d dataframes", len(dfs))
        dfs = stage(dfs)
        if args.debug_dfs:
            debug_df = dump_debug_df(dfs, args.debug_dfs_query)
            if debug_df is not None:
                logger.debug(debug_df.head(args.debug_rows).to_string())


if __name__ == "__main__":
    main()
