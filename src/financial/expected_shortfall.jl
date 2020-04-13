"""
    expected_shortfall(returns; risk_level::Real=0.05, per_MW=false, volumes=[]) -> Number

Calculate the expected shortfall `-𝔼[ r_p | r_p ≤ q_risk_level(r_p) ]`, where `r_p` is
the portfolio return and `q_risk_level(r_p)` is the lower quantile of the distribution
of `r_p` characterised by the `risk_level`.

If an insufficient number of `returns` is provided to calculate the expected shortfall
then this logs a warning and returns `missing`.

If `per_MW=true`, returns the average return of the bottom quantile divided by the average
volume of that quantile.

NOTE: Expected shortfall is the _negative_ of the average of the bottom quantile of
`return_samples`. Assuming average is positive for all `risk_level`, then it is good to
_minimise_ expected shortfall.

# Arguments
- `returns` (iterator): the portfolio of returns

# Keyword Arguments
- `risk_level::Real`: risk level associated with the lower quantile of the returns
distribution.
- `per_MW`: compute expected shortfall per MW.
- `volumes`: volumes used in case `per_MW=true`. Ignored otherwise.
"""
function expected_shortfall(returns; risk_level::Real=0.05, per_MW=false, volumes=[])
    0 < risk_level < 1 || throw(ArgumentError("risk_level=$risk_level is not between 0 and 1."))

    if per_MW == true && length(volumes) != length(returns)
        throw(ArgumentError("Need corresponding volumes in order to compute per MW."))
    end

    returns = collect(returns)
    last_index = floor(Int, risk_level * length(returns))
    if last_index === 0
        @warn(
            "Too few samples provided to calculate expected shortfall for given risk-level.",
             risk_level,
             minimum_number_of_samples=ceil(Int, 1/risk_level),
             number_of_samples=length(returns),
        )
        return missing
    end

    scale = per_MW ? mean(volumes[partialsortperm(returns, 1:last_index)]) : 1.0

    return valtype(returns)(-mean(partialsort(returns, 1:last_index)) / scale)
end

ObservationDims.obs_arrangement(::typeof(expected_shortfall)) = MatrixColsOfObs()
const es = expected_shortfall

"""
    expected_shortfall(
        volumes::AbstractVector, deltas::AbstractArray, args...; kwargs...,
    ) -> Number

Calculate the sample expected shortfall of the distribution of `returns` - with
[`price impact`](@ref price_impact) ``w'μ − w'Πw``, according to a certain `risk_level`,
given a portfolio of `volumes` and a sample of price `deltas`.

# Arguments
- `volumes::AbstractVector`: the portfolio of volumes
- `deltas::AbstractArray`: collection of price deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).

# Keyword Arguments
- `risk_level::Real`: risk level associated with the lower quantile of the returns distribution
"""
function expected_shortfall(
    volumes::AbstractVector, deltas::AbstractArray, args...; kwargs...,
)
    @assert length(args) < 3
    returns = _calculate_returns(volumes, deltas)
    pi = price_impact(volumes, args...)
    exp_shortfall = expected_shortfall(returns; kwargs...)

    # note the sign on pi is reversed because ES had a sign reversal as well
    return exp_shortfall + pi
end

"""
    expected_shortfall(returns::Normal; risk_level::Real=0.05) -> Number

Calculate the analytic expected shortfall of a Normal distribution of `returns` according to
a certain `risk_level`.
"""
function expected_shortfall(returns::Normal; risk_level::Real=0.05)
    # ES(w) = (p(q_{1-α}(r)) / α) * (w'Σw)^{1/2} − w'μ
    # See Section 6, https://drive.google.com/file/d/1SU03QYm-RRmyOKHR-Ap5OrZiqP1NiNr5/view
    # https://en.wikipedia.org/wiki/Expected_shortfall#Normal_distribution
    ϕ = pdf(Normal(), quantile(Normal(), risk_level))
    return (ϕ / risk_level) * std(returns) - mean(returns)
end

"""
    expected_shortfall(volumes::AbstractVector, deltas::Sampleable{Multivariate}, args...; kwargs...) -> Number

Calculate the analytic expected shortfall of the distribution of `returns` - with
[`price impact`](@ref price_impact) ``w'μ − w'Πw`` -  according to a certain `risk_level`
given a portfolio of `volumes` and known `distribution` of price deltas.

# Arguments
- `volumes::AbstractVector`: the portfolio of `volumes`
- `deltas::Sampleable{Multivariate}`: the joint distribution of the price deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).

# Keyword Arguments
- `kwargs::Real`: risk level associated with the lower quantile of the returns distribution
"""
function expected_shortfall(volumes::AbstractVector, deltas::Sampleable{Multivariate}, args...; kwargs...)
    @assert length(args) < 3
    mean_returns = expected_return(volumes, deltas, args...)
    sigma_returns = volatility(volumes, deltas)
    return_dist = Normal(mean_returns, sigma_returns)
    return expected_shortfall(return_dist; kwargs...)
end

"""
    median_over_expected_shortfall(returns::AbstractVector, args...; kwargs...) -> Number
    median_over_expected_shortfall(
        volumes::AbstractVector,
        deltas::AbstractMatrix,
        args...;
        kwargs...
    ) -> Number

Calculate the `median(returns) / expected_shortfall(returns)` metric, aka the `evano`
metric.

For the function that takes in `returns`, we must assume that the price impact has already
been included.

We currently don't have a working version of this for Multivariate Distributions as there
are many definitions of `median` which aren't implemented by `Distributions`.
https://invenia.slack.com/archives/CMMAKP97H/p1567612804011200?thread_ts=1567543537.008300&cid=CMMAKP97H
More info: https://www.r-bloggers.com/multivariate-medians/

# Arguments
- `returns::AbstractVector`: An iterator of returns over some time or of some portfolio
- `volumes::AbstractVector`: The MWs volumes of the portfolio
- `deltas::AbstractMatrix`: The sample of price deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).

# Keyword Arguments
- `kwargs`: The [`expected shortfall`](@ref expected_shortfall) keyword arguments.
"""
function median_over_expected_shortfall(returns; kwargs...)
    return median(returns) / expected_shortfall(returns; kwargs...)
end

function median_over_expected_shortfall(
    volumes::AbstractVector, deltas::AbstractMatrix, args...; kwargs...
)
    m_return = median_return(volumes, deltas, args...)
    es_return = expected_shortfall(volumes, deltas, args...; kwargs...)
    return m_return / es_return
end

ObservationDims.obs_arrangement(::typeof(median_over_expected_shortfall)) = MatrixColsOfObs()
const evano = median_over_expected_shortfall
