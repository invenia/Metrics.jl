# API

```@index
Modules = [Metrics]
```

## Evaluate
```@docs
evaluate
```

## Regression Metrics

```@docs
expected_squared_error
expected_absolute_error
mean_squared_error
root_mean_squared_error
normalised_root_mean_squared_error
standardized_mean_squared_error
mean_absolute_error
mean_absolute_scaled_error
prediction_interval_coverage_probability
window_prediction_interval_coverage_probability
adjusted_prediction_interval_coverage_probability
potential_payoff
marginal_gaussian_loglikelihood
joint_gaussian_loglikelihood
```

## Financial Metrics

```@docs
expected_return
volatility
sharpe_ratio
median_return
expected_shortfall
median_over_expected_shortfall
price_impact
```

## Model Metrics

```@docs
kullback_leibler
```

## Summary Functions
```@docs
regression_summary
financial_summary
```

## Implementation

For implementation details refer to the [Metrics.jl documentation](https://invenia.pages.invenia.ca/Metrics.jl/).
