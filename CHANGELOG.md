# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.2] - 2020-09-29
### Fixed
- Issue with uncertainty band on stacked plots, PR #45

## [0.8.1] - 2020-06-19
### Fixed
- Allow multiple expansions for KeepSpecificBins in postproc, PR #38

## [0.8.0] - 2020-05-22
### Added
- Useability tweaks for plotting

### Fixed
- Edge bins when labels are alphanumeric are now full-width (not half-width)

### Dropped
- No longer support python 2 (since we need Pandas >= 1.0)

## [0.7.0] - 2020-05-14
### Added
- Post-processing functions that interact with curator to apply or rebin datasets, PR #36 [@benkrikler](https://github.com/benkrikler)

## [0.6.5] - 2020-05-12
### Added
- Implement the multiply_values with a mapping, PR #35 [@benkrikler](https://github.com/benkrikler)

## [0.6.4] - 2020-05-07
### Added
- New postprocessing stage to filter columns, PR #34 [@benkrikler](https://github.com/benkrikler)
- New option to AssignCols stage to make assignment the index, PR #34

## Fixed
- ToDatacardInputs had been broken in a previous update, PR #34

## [0.6.3] - 2020-04-29
### Added
- Add ability to rotate x-axis tick labels from config
- Add GenericPandas and UnstackWeights stages, PR #33 [@benkrikler](https://github.com/benkrikler)

## [0.6.2] - 2020-04-21
### Fixed
- Fix `split` function in postproc module to work with numbers, PR #32 [@benkrikler](github.com/benkrikler)

## [0.6.1] - 2020-04-20
### Added
- Option to specify which columns contain bin values rather than labels, PR #31 [@benkrikler](github.com/benkrikler)

### Fixed
- Report package version number properly, PR #31

## [0.6.0] - 2020-04-17
### Added
- Proper support for non-constant interval widths

## [0.5.1] - 2020-04-7
### Fixed
- Bugs in post-processing modules, PR #29 [@benkrikler](github.com/benkrikler)

## [0.5.0] - 2020-03-29
### Added
- New post-processing command to reshape outputs of fast-carpenter from PR #28 [@benkrikler](github.com/benkrikler)

## [0.4.0] - 2020-02-26
- Many changes from PR #26 [@benkrikler](github.com/benkrikler)

### Fixed
- Bugs in the way overflow bins were handled and step-lines were drawn by padding. Impacted error bars as well as produced weird plotting artefacts.

### Added 
- Extend unit tests
- Variable interpolation within the config files and using variables which can be passed from the command-line
- Y-limits based on plot-margins: pass a float with a percent sign after to limit configs
- Control over the display of under and overflow bins from the config file
- Ability to give specific colours for individual bands in the plot
- Option to control how errors are calculated: sqrt of sumw2 or sumw / sqrt(n)

## [0.3.0] - 2019-11-1
- Many changes from PR #13 [@benkrikler](github.com/benkrikler)
### Added 
- Error bands in ratio plot on the expected R=1 line and the error markers
- Control over the colour pallete from the config file
- Control over the figure size
- News command-line options to make multiple image types (e.g. pdf and png) from a single command invocation
- This CHANGELOG was added

### Changed
- Datasets can now be ordered, which only affects the colouring; they're still stacked according to total integral
- Fixes for using weighted events for data: use the dataset_col and data regex, as opposed to checking for NaNs

## [0.2.1] - 2019-5-1
## [0.2.0] - 2019-3-27
## [0.1.1] - 2019-3-1
