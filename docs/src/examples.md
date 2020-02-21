### Basic Use

The recommended way to calculate a metric is to use [`evaluate`](@ref), e.g. `evaluate(metric, args...)`, since this takes the format of the data into account using the `ObsArrangement`.

For example, it will wrangle any matrix data into the correct format, which avoids the confusion around whether `observations` are rows or columns.

```@example evaluate
using Metrics
volumes = rand(10);
deltas = rand(5, 10);

# columns are observations
evaluate(expected_return, volumes, deltas)
```

You can also pass in an optional `obsdim` argument specifying what dimension your observations is on (`1=row`, `2=column`)).

```@example evaluate
deltas_2 = rand(10, 5);

# rows are observations
evaluate(expected_return, volumes, deltas_2; obsdim=2)
```

If not provided, it will work it out with sensible handling for `NamedDimsArray`s, and for generic arrays it will fall back to assuming `row`s.
For some metrics, which are `SingleObs`, this argument is ignored.

One could always just invoke the metric normally, e.g. `metric(args...)`

```@example evaluate
expected_return(volumes, deltas_2)
```

but would need to ensure the data is oriented correctly otherwise it will error.


### IndexedDistributions and AxisArrays

Any `regression` metric compatible with `IndexedDistribution`s and `AxisArray`s will match their indices to ensure the metric is computed correctly, even outside of [`evaluate`](@ref),

```@example
using Metrics, Distributions, IndexedDistributions, AxisArrays

predictions = IndexedDistribution(MvNormal([1, 2, 3], ones(3)), ["a", "b", "c"])
truths = AxisArray([1, 2, 3], Axis{:nodes}(["b", "c", "a"]))

evaluate(mse, truths, predictions)
```

### Summary functions

All of the `regression` and `financial` metrics can be computed in one function call using the [`regression_summary`](@ref) and [`financial_summary`](@ref) functions, respectively.

The following metrics are calculated in each summary
* `regression`: [`mean_squared_error`](@ref), [`root_mean_squared_error`](@ref), [`normalised_root_mean_squared_error`](@ref), [`standardized_mean_squared_error`](@ref), [`mean_absolute_error`](@ref), [`potential_payoff`](@ref).
* `financial`: [`expected_return`](@ref), [`expected_shortfall`](@ref), [`sharpe_ratio`](@ref), [`volatility`](@ref) for PO evaluation, where [`median_over_expected_shortfall`](@ref) and [`median_return`](@ref) are also computed for financial return evaluation.

The summary functions make calculating these metrics much easier and efficient and are particularly useful in backruns and hyper-parameter tuning jobs.
As with other metrics, summary functions are also compatible with [`evaluate`](@ref).

```@example summary
using Metrics
predictions = rand(10);
truths = rand(10);

evaluate(regression_summary, truths, predictions)
```