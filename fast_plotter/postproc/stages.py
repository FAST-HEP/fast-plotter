from . import functions


class BaseManipulator():
    def __init__(self, **kwargs):
        use_outdir = getattr(self, "use_outdir", False)
        if not use_outdir:
            kwargs.pop("out_dir", None)
        self.name = kwargs.pop("name")
        self.kwargs = kwargs
        self.func = getattr(functions, self.func)

    def __call__(self, dfs):
        if self.cardinality == "many-to-one":
            out = [self.func(dfs, **self.kwargs)]
        elif self.cardinality == "one-to-many":
            out = []
            for df in dfs:
                out += self.func(df, **self.kwargs)
        elif self.cardinality == "one-to-one":
            out = [self.func(df, **self.kwargs) for df in dfs]
        elif self.cardinality == "none-to-many":
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
