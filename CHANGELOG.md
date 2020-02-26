# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed

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
