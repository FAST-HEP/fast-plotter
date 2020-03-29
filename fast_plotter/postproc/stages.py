from . import functions


def _unique_vals(entries):
    from collections import defaultdict
    unique = defaultdict(set)
    for entry in entries:
        for key, val in entry.items():
            unique[key].add(val)
    unique = {k: tuple(v) if len(v) > 1 else v.pop() for k, v in unique.items()}
    return unique


class BaseManipulator():
    give_meta = False

    def __init__(self, **kwargs):
        use_outdir = getattr(self, "use_outdir", False)
        if not use_outdir:
            kwargs.pop("out_dir", None)
        self.name = kwargs.pop("name")
        self.kwargs = kwargs
        self.func = getattr(functions, self.func)
        self.doc = self.func.__doc__

    def __call__(self, dfs):
        if self.cardinality == "many-to-one":
            meta = _unique_vals([d[1] for d in dfs])
            dfs = [d[0] for d in dfs]
            out = [(self.func(dfs, **self.kwargs), meta)]
        elif self.cardinality == "one-to-many":
            self.kwargs.setdefault("return_meta", True)
            out = []
            for df, meta in dfs:
                results = self.func(df, **self.kwargs)
                results = [(df, _unique_vals([meta, m])) for df, m in results]
                out += results
        elif self.cardinality == "one-to-one":
            if self.give_meta:
                out = [(self.func(df, meta=m, **self.kwargs), m) for df, m in dfs]
            else:
                out = [(self.func(df, **self.kwargs), m) for df, m in dfs]
        elif self.cardinality == "none-to-many":
            self.kwargs.setdefault("return_meta", True)
            out = dfs + self.func(**self.kwargs)
        return out


class Query(BaseManipulator):
    cardinality = "one-to-one"
    func = "query"


class ReBin(BaseManipulator):
    cardinality = "one-to-one"
    func = "rebin"


class KeepSpecificBins(BaseManipulator):
    cardinality = "one-to-one"
    func = "keep_specific_bins"


class CombineColumns(BaseManipulator):
    cardinality = "one-to-one"
    func = "combine_cols"


class RegexSplitDimension(BaseManipulator):
    cardinality = "one-to-one"
    func = "regex_split_dimension"


class RenameCols(BaseManipulator):
    cardinality = "one-to-one"
    func = "rename_cols"


class RenameBinningDimension(BaseManipulator):
    cardinality = "one-to-one"
    func = "rename_dim"


class Split(BaseManipulator):
    cardinality = "one-to-many"
    func = "split"


class ReorderDimensions(BaseManipulator):
    cardinality = "one-to-one"
    func = "reorder_dimensions"


class Densify(BaseManipulator):
    cardinality = "one-to-one"
    func = "densify"


class StackWeights(BaseManipulator):
    cardinality = "one-to-one"
    func = "stack_weights"


class ToDatacardInputs(BaseManipulator):
    cardinality = "one-to-one"
    func = "to_datacard_inputs"


class AssignCol(BaseManipulator):
    cardinality = "one-to-one"
    func = "assign_col"


class AssignDim(BaseManipulator):
    cardinality = "one-to-one"
    func = "assign_dim"


class Merge(BaseManipulator):
    cardinality = "many-to-one"
    func = "merge"


class MultiplyValues(BaseManipulator):
    cardinality = "one-to-one"
    func = "multiply_values"


class NormaliseGroup(BaseManipulator):
    cardinality = "one-to-one"
    func = "normalise_group"


class OpenMany(BaseManipulator):
    cardinality = "none-to-many"
    func = "open_many"


class WriteOut(BaseManipulator):
    cardinality = "one-to-one"
    func = "write_out"
    use_outdir = True
    give_meta = True
