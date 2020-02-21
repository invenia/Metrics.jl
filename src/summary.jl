const REGRESSION_METRICS = (mse, rmse, nrmse, smse, mae, potential_payoff)

"""
    regression_summary(y_true, y_pred, args...)

Calculate a summary of; @ref[`mean_squared_error`], @ref[`root_mean_squared_error`],
@ref[`normalised_root_mean_squared_error`], @ref[`standardized_mean_squared_error`],
@ref[`expected_absolute_error`], @ref[`mean_absolute_error`].

Returns a `NamedTuple` of metric names and results.
"""
regression_summary(args...) = (; (nameof(f) => f(args...) for f in REGRESSION_METRICS)...)
obs_arrangement(::typeof(regression_summary)) = SingleObs()



"""
    financial_summary(returns, args...; kwargs...)
    financial_summary(volumes::AbstractVector, deltas::Union{MvNormal, AbstractMatrix}, args...; kwargs...)

Calculate the summary of applicable financial metrics.
`args...` and `kwargs...` are inputs for the functions above.

# Keywords
- `per_mwh::Bool=false`: Scales financial metrics by per MWh instead by total volume.

Returns a `NamedTuple` of metric names and results.
"""
function financial_summary(
    volumes::AbstractVector, deltas::Union{MvNormal, AbstractMatrix}, args...;
    per_mwh::Bool=false, kwargs...
)
    scale = per_mwh ? 1 / sum(abs, volumes) : one(eltype(volumes))

    return (;
        :expected_return => scale * expected_return(volumes, deltas, args...),
        :expected_shortfall => scale * expected_shortfall(volumes, deltas, args...; kwargs...),
        :sharpe_ratio => sharpe_ratio(volumes, deltas, args...),
        :volatility => scale * volatility(volumes, deltas),
    )
end

function financial_summary(returns; risk_level::Real=0.05)
    return (;
        :median_return => median(returns),
        :expected_return => expected_return(returns),
        :expected_shortfall => expected_shortfall(returns; risk_level=risk_level),
        :median_over_expected_shortfall => median_over_expected_shortfall(returns; risk_level=risk_level),
        :sharpe_ratio => sharpe_ratio(returns),
        :volatility => volatility(returns),
    )
end