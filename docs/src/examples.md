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


### KeyedDistributions and KeyedArrays

Any `regression` metric compatible with `KeyedDistribution`s and `KeyedArray`s will match their indices to ensure the metric is computed correctly, even outside of [`evaluate`](@ref),

```@example
using Metrics, Distributions, KeyedDistributions, AxisKeys

predictions = KeyedDistribution(MvNormal([1, 2, 3], ones(3)), ["a", "b", "c"])
truths = KeyedArray([1, 2, 3]; nodes=["b", "c", "a"])

evaluate(mse, truths, predictions)
```

### Summary functions

All of the `regression` and `financial` metrics can be computed in one function call using the [`regression_summary`](@ref) and [`financial_summary`](@ref) functions, respectively.


The summary functions make calculating these metrics much easier and efficient and are particularly useful in backruns and hyper-parameter tuning jobs.
As with other metrics, summary functions are also compatible with [`evaluate`](@ref).

```@example summary
using Metrics
predictions = rand(10);
truths = rand(10);

evaluate(regression_summary, truths, predictions)
```
