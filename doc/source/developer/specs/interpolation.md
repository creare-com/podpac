# Requirements
TODO: Add requirements

# Example Use cases
TODO: Add example usecases

# Specification

## Interpolator Class

#### Constants

- `interpolate_options`:
    - Enum('nearest', 'nearest_preview', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3'), 
    - Default: `nearest`
    - Only include the supported interpolation options

#### Traits

- `method`: one of:
    - str: Enum(`interpolate_options`)
    - Dict({`dim`: Enum(`interpolate_options`)})
    - For all dims or single dims.

## User Interface

TODO: Add user interface specs

## Developer interface 
TODO: Add developer interface specs
