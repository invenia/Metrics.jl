const REGRESSION_METRICS = (mse, rmse, nrmse, smse, mae, potential_payoff)

"""
    regression_summary(y_true, y_pred, args...)

Calculate a summary of; @ref[`mean_squared_error`], @ref[`root_mean_squared_error`],
@ref[`normalised_root_mean_squared_error`], @ref[`standardized_mean_squared_error`],
@ref[`expected_absolute_error`], @ref[`mean_absolute_error`].

Returns a `NamedTuple` of metric names and results.
"""
regression_summary(args...) = (; (nameof(f) => f(args...) for f in REGRESSION_METRICS)...)
ObservationDims.obs_arrangement(::typeof(regression_summary)) = SingleObs()



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

function financial_summary(returns; risk_level::Real=0.05, kwargs...)
    return (;
        :median_return => median(returns),
        :expected_return => expected_return(returns),
        :expected_shortfall => expected_shortfall(returns; risk_level=risk_level, kwargs...),
        :expected_windfall => expected_windfall(returns; level=risk_level, kwargs...),
        :median_over_expected_shortfall => median_over_es(returns; risk_level=risk_level, kwargs...),
        :sharpe_ratio => sharpe_ratio(returns),
        :volatility => volatility(returns),
        :mean_over_expected_shortfall => mean_over_es(returns; risk_level=risk_level, kwargs...),
    )
end

"""
    financial_summary(
        returns, volumes;
        risk_level::Real=0.05,
        per_mwh=false,
    ) -> NamedTuple

Calculate and stores basic financial data about a set of returns and volumes.
Values that cannot be calculated will be missing.

Volume will be absolute valued before stats are calculated.

If `per_mwh=true`, notice that `total_return` becomes a quantity with no practical meaning,
as it is just the sum of daily return per MWh fractions.

# Arguments
- `returns`: Financial returns.
- `volumes`: Volume associated with the same index of financial return.

# Keyword Arguments
- `risk_level::Real=0.05`: The risk level for expected shortfall. For the default, the returns
must be in the bottom 5 percent of returns.
- `per_mwh=false`: Compute quantities per MWh.

# Returns
- `NamedTuple`: Calculated basic financial statistics.
    Keys of the returned dictionary:
    :total_volume - Total volume traded
    :mean_volume - Mean volume traded of a period
    :median_volume - Median volume traded of a period
    :std_volume - standard deviation of volume traded each period
    :total_return - Total financial return
    :mean_return - Mean financial return of a period
    :median_return - Median financial return of a period
    :std_return - standard deviation of the period financial returns
    :expected_shortfall - expected shortfall of returns i.e. -E[r | r <= q_α(r)], q_α is the
    α-percentile
    :expected_windfall - expected windfall of returns.
    :mean_over_expected_shortfall - Mean return divided by expected shortfall
    :median_over_expected_shortfall - Median return divided by expected shortfall
    """
function financial_summary(
    returns, volumes;
    risk_level::Real=0.05,
    per_mwh=false,
)
    # We expect these to be equal in size
    if length(returns) != length(volumes)
        error(
            """
            Given return and volume vectors must be the same length.
            Return length: $(length(returns))
            Volume length: $(length(volumes))
            """
        )
    end

    # Get rid of missing values
    # We want to get rid of values that have a missing in either the volume or
    # the returns vector, so we can't just use skipmissing
    # Note: `disallowmissing` converts the type of the array to no longer allow `missing`
    # values. We do this to ensure that `std` on `FixedDecimal` arrays always works.
    not_missing_mask = .!(ismissing.(returns) .| ismissing.(volumes))
    returns = disallowmissing(returns[not_missing_mask])
    volumes = disallowmissing(volumes[not_missing_mask])

    # Make everything missing if the return vector is empty, except traded periods which
    # we know the value of correctly.
    # Return the stats dict here so we don't try to do any work on empty array.

    all_missing = (;
        :total_volume => missing,
        :mean_volume => missing,
        :median_volume => missing,
        :std_volume => missing,
        :total_return => missing,
        :mean_return => missing,
        :median_return => missing,
        :std_return => missing,
        :expected_shortfall => missing,
        :expected_windfall => missing,
        :mean_over_expected_shortfall => missing,
        :median_over_expected_shortfall => missing,
    )

    isempty(returns) && return all_missing

    # Take the absolute of volumes, we don't want our volume negating itself here.
    # We want the actual total.
    volumes = abs.(volumes)

    # Determine scaling factor
    # Calling `float()` here because statistics over money should not have precision
    # only up to cents, thus we have to remove the `FixedDecimal`s.
    scale = per_mwh ? convert.(Float64, volumes) : ones(length(returns))

    # Not using scale directly below because the total dollar is also dollar.
    # Note that sum(returns ./ scale) is not an extensive property
    # and as a metric it is not grounded in any practical principle,
    # yet it can be used heuristically in some analyses if regarded with scrutiny.
    total_return = per_mwh ? sum(returns) : sum(returns ./ scale)

    fin_sum = financial_summary(
        float.(returns);
        risk_level=risk_level,
        per_mwh=per_mwh,
        volumes=volumes,
    )

    # We cannot use the mean and median from the other financial output
    # since it doesn't account for scale so we must compute and store those:
    mean_return = mean(returns) / mean(scale)
    # Notice that here we are computing the median of the ratios, not the ratio of the
    # medians. For discussion, see https://gitlab.invenia.ca/invenia/Metrics.jl/-/issues/56
    median_return = NaNMath.median(returns ./ scale)

    # We now have everything we need to build up our stats dictionary
    return (;
        :total_volume => sum(volumes),
        # Calling `float()` here because statistics over money should not have precision
        # only up to cents, thus we have to remove the `FixedDecimal`s.
        :mean_volume => mean(float.(volumes)),
        :median_volume => median(float.(volumes)),
        # Calculating the `std` when we have ≤ 1 FixedDecimal values will result in
        # attempting to convert a `NaN` into a FixedDecimal which will error. As a work
        # around we'll use `missing` in these cases.
        # See: https://github.com/JuliaLang/julia/issues/25300
        :std_volume => length(volumes) <= 1 ? missing : std(volumes),
        :total_return => total_return,
        :mean_return => mean_return,
        :median_return => median_return,
        # take results from the other financial_summary method
        :std_return => length(returns) <= 1 ? missing : std(returns) / mean(scale),
        :expected_shortfall => fin_sum.expected_shortfall,
        :expected_windfall => fin_sum.expected_windfall,
        :mean_over_expected_shortfall => mean_return / fin_sum.expected_shortfall,
        :median_over_expected_shortfall => median_return / fin_sum.expected_shortfall,
    )
end
