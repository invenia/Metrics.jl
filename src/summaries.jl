const REGRESSION_METRICS = (mse, rmse, nrmse, smse, mae)

"""
    regression_summary(y_true, y_pred, args...)

Calculate a summary of; @ref[`mean_squared_error`], @ref[`root_mean_squared_error`],
@ref[`normalised_root_mean_squared_error`], @ref[`standardized_mean_squared_error`],
@ref[`expected_absolute_error`], @ref[`mean_absolute_error`].

Returns a Dictionary where the `Key` is the function, and the `Value` is the result of the function.
"""
function regression_summary(args...)
    summary = Dict()

    for metric in REGRESSION_METRICS
        summary[metric] = metric(args...)
    end

    return summary
end
obs_arrangement(::typeof(regression_summary)) = SingleObs()



"""
    financial_summary(returns, args...; kwargs...)
    financial_summary(volume::AbstractArray, deltas::Union{MvNormal, AbstractMatrix}, args...; kwargs...)

Calculate the summary of applicable financial metrics.
`args...` and `kwargs...` are inputs for the functions above.

Returns a Dictionary where the `Key` is the function, and the `Value` is the result of the function.
"""
function financial_summary(
    volumes::AbstractArray, deltas::Union{MvNormal, AbstractMatrix}, args...;
    kwargs...
)
    return Dict(
        expected_return => expected_return(volumes, deltas, args...),
        expected_shortfall => expected_shortfall(volumes, deltas, args...; kwargs...),
        sharpe_ratio => sharpe_ratio(volumes, deltas, args...),
        volatility => volatility(volumes, deltas)
    )
end

function financial_summary(returns; risk_level::Real=0.05)
    return Dict(
        median_return => median(returns),
        expected_return => expected_return(returns),
        expected_shortfall => expected_shortfall(returns; risk_level=risk_level),
        median_over_expected_shortfall => median_over_expected_shortfall(returns; risk_level=risk_level),
        sharpe_ratio => sharpe_ratio(returns),
        volatility => volatility(returns),
    )
end
