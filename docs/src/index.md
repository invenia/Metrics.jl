# Metrics.jl


## Why does this package exist:
The purpose of `Metrics` is to provide a one-stop-shop for a variety of functions we might use when assessing
 - Accuracy of models and Forecasters
 - Financial returns
 - Comparing models

As we uncover more use cases for new metrics, they should be added to this package under a new category if necessary.

## Metric Categories

Metrics are organized according to their typical use-cases: `regression`, `model`, and `financial`.

The `regression` metrics are used to evaluate the accuracy of a model's `predicted` values against some known `truth`.
Since our [`BaselineForecasters`](https://gitlab.invenia.ca/invenia/BaselineForecasters.jl) and [`GPForecaster`](https://gitlab.invenia.ca/research/GPForecaster.jl) predict posterior distributions over the target variables (delta LMPs), the `regression` metrics are also compatible with [`Distributions`](https://github.com/JuliaStats/Distributions.jl), as well as [`KeyedDistributions`](https://github.com/invenia/KeyedDistributions.jl) and [`AxisKeys`](https://github.com/mcabbott/AxisKeys.jl).
Thus, they may be used at all levels of the [RTE pipeline](https://gitlab.invenia.ca/invenia/wiki/blob/master/research/research-testing-environment.md#design) to evaluate the accuracy of a model or [`Forecaster`](https://gitlab.invenia.ca/invenia/Forecasters.jl), from the experimentation phase (green) up to full backruns (red).

The `model` metrics are used to compare the outputs of models and forecasters against each other.
Unlike `regression` metrics, they do not compare `prediction`s against a known `truth`, hence they form a separate category.

The `financial` metrics are used in evaluating the outputs of [Portfolio Optimization](https://gitlab.invenia.ca/invenia/PortfolioOptimizers.jl) or the returns of an extended backrun after the [`financials`](https://invenia.pages.invenia.ca/BidFinance.jl/pages/api/#BidFinance.get_financials-Tuple{S3DB.AbstractClient,TimeZones.ZonedDateTime,Bids.FixedDataFrame,ElectricityMarkets.Market}) have been calculated.
These contain functions for computing quantities such as `median_return`, `sharpe_ratio`, and `expected_shortfall`.
This would typically be used during the [`RiskAnalysis`](https://gitlab.invenia.ca/invenia/RiskAnalysis.jl) performed on a backrun.

## How does Metrics work?

### Traits and `evaluate`

As with [`BaselineModels`](https://gitlab.invenia.ca/research/BaselineModels.jl) this package uses [traits](https://white.ucc.asn.au/2018/10/03/Dispatch,-Traits-and-Metaprogramming-Over-Reflection.html).

The only trait in current use is the `ObsArrangement`, which is imported from [ObservationDims.jl](https://github.com/invenia/ObservationDims.jl) and specifies the arrangement a metric expects the provided data to be in.
This is mostly taken care of by the [`evaluate`](@ref) function.
