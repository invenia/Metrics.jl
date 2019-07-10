# Metrics

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://invenia.pages.invenia.ca/Metrics.jl/)
[![Build Status](https://gitlab.invenia.ca/invenia/Metrics.jl/badges/master/build.svg)](https://gitlab.invenia.ca/invenia/Metrics.jl/commits/master)
[![Coverage](https://gitlab.invenia.ca/invenia/Metrics.jl/badges/master/coverage.svg)](https://gitlab.invenia.ca/invenia/Metrics.jl/commits/master)


## Short how to use:
 - this package contains metrics
 - the normal way to call a metric for is `evaluate(metric, args...)`
     - this takes into account making sure your data is in the right form
     - you can pass in an `obsdim` argument specifying what dimension your observations is on (1=row, 2=column)), but it is optional, and it will work it out for iterators of observations, `NamedDimsArray`s, and for generic arrays fall back to assuming rows.
     - or for some metrics you don't have multile observation e.g `squared_error`, it is ignored.
 - if you are trying to max out performance, then you can call the metric directly, if you sort out your observations youself to match what it expects, though that will basically have the same performance as calling `evaluate` with the observations already in the right form.
 
