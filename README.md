# Metrics

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://invenia.pages.invenia.ca/Metrics.jl/)
[![Build Status](https://gitlab.invenia.ca/invenia/Metrics.jl/badges/master/build.svg)](https://gitlab.invenia.ca/invenia/Metrics.jl/commits/master)
[![Coverage](https://gitlab.invenia.ca/invenia/Metrics.jl/badges/master/coverage.svg)](https://gitlab.invenia.ca/invenia/Metrics.jl/commits/master)

## Why does this package exists:
 - This package holds evaluation metrics for a variety of uses including assessing forecasts and financials
 - It consolidates metrics that previously were in spread though out various expermental scripts, analysis packages, and functional packages
 - The metrics here are designed to be perforant and consistent to a single definition.
 - If one finds oneself writing a function designed for the evaluation of something, odds are it belongs in the package.

## Short how to use:
 - the normal way to call a metric for is `evaluate(metric, args...)`
     - this takes into account making sure your data is in the right form
     - you can pass in an `obsdim` argument specifying what dimension your observations is on (1=row, 2=column)), but it is optional
     - If not provided it will work it out with sensible handling for `NamedDimsArray`s, and for generic arrays fall back to assuming rows.
     - or for some metrics you don't have multile observation e.g `squared_error`, it is ignored.
 - if you are trying to max out performance, then you can call the metric directly.
    - but then you must ensure yourself that the observations are in the right form,
    - and if you do this then you will also see the same performance in calling `evaluate` with this arrangement.
    - but still you might prefer calling directly as that will likely error if your arrangement is not what you think
