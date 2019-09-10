
"""
    expected_return(volumes::AbstractVector, deltas::AbstractVector, args...) -> Number
    expected_return(volumes::AbstractVector, deltas::AbstractMatrix, args...) -> Number
    expected_return(volumes::AbstractVector, deltas::MvNormal, args...) -> Number

Calculate the expected mean return with [`price impact`](@ref price_impact) ``w'μ − w'Πw``
of a portfolio of `volumes` given a distribution, sample, or vector of price `deltas`.

## Arguments
- `volumes::AbstractVector`: The MWs volumes of the portfolio
- `deltas`: The collection of prices deltas which can be a `MvNormal`, `AbstractMatrix`, or
    `AbstractVector`
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).
"""
function expected_return(volumes::AbstractVector, deltas::AbstractVector, args...)
    @assert length(args) < 3
    exp_return = dot(volumes, deltas)
    pi = price_impact(volumes, args...)
    return exp_return - pi
end

function expected_return(volumes::AbstractVector, deltas::AbstractMatrix, args...)
    volumes = NamedDimsArray(volumes, :nodes)
    deltas = NamedDimsArray(deltas, (:nodes, :obs))
    expected_deltas = vec(mean(deltas, dims=:obs))
    return  expected_return(volumes, expected_deltas, args...)
end

function expected_return(volumes::AbstractVector, deltas::MvNormal, args...)
    return expected_return(volumes, mean(deltas), args...)
end

obs_arrangement(::typeof(expected_return)) = MatrixColsOfObs()


"""
    volatility(volumes::AbstractVector, deltas::AbstractMatrix) -> Number
    volatility(volumes::AbstractVector, deltas::MvNormal) -> Number

Calculate the expected standard deviation of returns ``(w'Σw)^{1/2}``.
"""
function volatility(volumes::AbstractVector, deltas::AbstractMatrix)
    volumes = NamedDimsArray(volumes, :nodes)
    deltas = NamedDimsArray(deltas, (:nodes, :obs))
    vol = std(deltas' * volumes; dims=:obs)
    return first(vol)  # vol is a 1-element NamedDimsArray hence first()
end

function volatility(volumes::AbstractVector, deltas::MvNormal)
    # Note, although `sqrtcov(deltas)` is in size `variable_length * node`,
    # `sqrtcov(deltas) * volumes` is a vector of length `variable_length`.
    # Taking the L2 norm of it, it becomes a scalar.
    # In short, although `sqrtcov(deltas)` is not unique, the `volatility`
    # function is calculating a unique scalar: `sqrt(volume' * cov(delta) * volume)`
    return norm(sqrtcov(deltas) * volumes, 2)
end

obs_arrangement(::typeof(volatility)) = MatrixColsOfObs()

"""
    sharpe_ratio(returns::AbstractVector) -> Number
    sharpe_ratio(
        volumes::AbstractVector, deltas::Union{MvNormal, AbstractMatrix}, args...,
    ) -> Number

Calculate the sharpe ratio, as the [`expected return`](@ref expected_return) over the
[`expected volatility`](@ref volatility) where the [`price impact`](@ref price_impact) is
also supported.

Formally, the sharpe_ratio is defined between two portfolios but we use the baseline
portfolio of _no-bid_ in this metric.

## Arguments
- `returns::AbstractVector`: the portfolio of returns
- 'volumes::AbstractVector': The MWs volumes of the portfolio
- `deltas::Union{MvNormal, AbstractMatrix}`: The distribution or sample of price deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).
"""
function sharpe_ratio(
    volumes::AbstractVector, deltas::Union{MvNormal, AbstractMatrix}, args...,
)
    mean_return = expected_return(volumes, deltas, args...)
    std_return = volatility(volumes, deltas)
    return mean_return / std_return
end

sharpe_ratio(returns) =  mean(returns) / std(returns)

obs_arrangement(::typeof(sharpe_ratio)) = MatrixColsOfObs()


"""
    median_return(volumes::AbstractVector, deltas::AbstractMatrix, args...) -> Number

Calculate the median return of a portfolio of `volumes` given a sample of price `deltas`.

## Arguments
- `volumes::AbstractVector`: The MWs volumes of the portfolio
- `deltas`::`AbstractMatrix`: The collection of prices deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).
"""
function median_return(volumes::AbstractVector, deltas::AbstractMatrix, args...)
    volumes = NamedDimsArray(volumes, :nodes)
    deltas = NamedDimsArray(deltas, (:nodes, :obs))
    returns = deltas' * volumes
    pi = price_impact(volumes, args...)

    return median(returns) - pi
end

obs_arrangement(::typeof(median_return)) = MatrixColsOfObs()

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
- `returns::AbstractVector`: A vector of returns over some time or of some portfolio
- `volumes::AbstractVector`: The MWs volumes of the portfolio
- `deltas::AbstractMatrix`: The sample of price deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).

# Keyword Arguments
- `kwargs`: The [`expected shortfall`](@ref expected_shortfall) keyword arguments.
"""
function median_over_expected_shortfall(returns::AbstractVector; kwargs...)
    return median(returns) / expected_shortfall(returns; kwargs...)
end

function median_over_expected_shortfall(
    volumes::AbstractVector, deltas::AbstractMatrix, args...; kwargs...
)
    m_return = median_return(volumes, deltas, args...)
    es_return = expected_shortfall(volumes, deltas, args...; kwargs...)
    return m_return / es_return
end

obs_arrangement(::typeof(median_over_expected_shortfall)) = MatrixColsOfObs()
const evano = median_over_expected_shortfall


"""
    expected_shortfall(returns::AbstractVector; risk_level::Real) -> Number

Calculate the expected shortfall `-𝔼[ r_p | r_p ≤ q_risk_level(r_p) ]`, where `r_p` is
the portfolio return and `q_risk_level(r_p)` is the lower quantile of the distribution
of `r_p` characterised by the `risk_level`.

NOTE: Expected shortfall is the _negative_ of the average of the bottom quantile of
`return_samples`. Assuming average is positive for all `risk_level`, then it is good to
_minimise_ expected shortfall.

# Arguments
- `returns::AbstractVector`: the portfolio of returns

# Keyword Arguments
- `risk_level::Real`: risk level associated with the lower quantile of the returns
distribution

"""
function expected_shortfall(returns::AbstractVector; risk_level::Real=0.05)

    0 < risk_level < 1 || throw(ArgumentError("risk_level=$risk_level is not between 0 and 1."))

    last_index = floor(Int, risk_level * length(returns))
    last_index > 0 || throw(
        ArgumentError(string(
                "length(returns)=$(length(returns)) too few elements for risk_level=$risk_level.",
                " Min length(returns)=$(ceil(Int, 1/risk_level))",
        ))
    )

    return -mean(partialsort(returns, 1:last_index))
end

obs_arrangement(::typeof(expected_shortfall)) = MatrixColsOfObs()
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
    volumes = NamedDimsArray(volumes, :nodes)
    deltas = NamedDimsArray(deltas, (:nodes, :obs))
    returns = deltas' * volumes

    # calculate price impact
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
    expected_shortfall(volumes::AbstractVector, deltas::MvNormal, args...; kwargs...) -> Number

Calculate the analytic expected shortfall of the distribution of `returns` - with
[`price impact`](@ref price_impact) ``w'μ − w'Πw`` -  according to a certain `risk_level`
given a portfolio of `volumes` and known `distribution` of price deltas.

# Arguments
- `volumes::AbstractVector`: the portfolio of `volumes`
- `deltas::MvNormal`: the joint distribution of the price deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).

# Keyword Arguments
- `kwargs::Real`: risk level associated with the lower quantile of the returns distribution
"""
function expected_shortfall(volumes::AbstractVector, deltas::MvNormal, args...; kwargs...)
    @assert length(args) < 3
    mean_returns = expected_return(volumes, deltas, args...)
    sigma_returns = volatility(volumes, deltas)
    return_dist = Normal(mean_returns, sigma_returns)
    return expected_shortfall(return_dist; kwargs...)
end
