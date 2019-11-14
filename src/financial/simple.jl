"""
    _calculate_returns(volumes::AbstractVector, deltas::AbstractMatrix)

Calculate the returns from the given `volumes` and `deltas`.

- `volumes::AbstractVector`: The MWs volumes of the portfolio, one volume per node
- `deltas::AbstractMatrix`: The collection of prices deltas, expected to have dimensions
    of nodes × observations

Returns:
 - `returns::NamedDimsArray{(:obs,)}`: a vector of returns, one per observation.
"""
function _calculate_returns(volumes::AbstractVector, deltas::AbstractMatrix)
    # we put these in NamedDimsArrays as excutable documentation of what we expect the arrangement to represent
    # and so that we get back something with namedims
    volumes = NamedDimsArray(volumes, :nodes)
    deltas = NamedDimsArray(deltas, (:nodes, :obs))
    return  deltas' * volumes
end

"""
    expected_return(volumes::AbstractVector, deltas::AbstractVector, args...) -> Number
    expected_return(volumes::AbstractVector, deltas::AbstractMatrix, args...) -> Number
    expected_return(volumes::AbstractVector, deltas::Sampleable{Multivariate}, args...) -> Number

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
    returns = _calculate_returns(volumes, deltas)
    volumes = NamedDimsArray(volumes, :nodes)
    deltas = NamedDimsArray(deltas, (:nodes, :obs))
    expected_deltas = vec(mean(deltas, dims=:obs))
    return  expected_return(volumes, expected_deltas, args...)
end

function expected_return(volumes::AbstractVector, deltas::Sampleable{Multivariate}, args...)
    return expected_return(volumes, mean(deltas), args...)
end
expected_return(returns) = mean(returns)
obs_arrangement(::typeof(expected_return)) = MatrixColsOfObs()


"""
    volatility(volumes::AbstractVector, deltas::AbstractMatrix) -> Number
    volatility(volumes::AbstractVector, deltas::Sampleable{Multivariate}) -> Number

Calculate the expected standard deviation of returns ``(w'Σw)^{1/2}``.
"""
function volatility(volumes::AbstractVector, deltas::AbstractMatrix)
    returns = _calculate_returns(volumes, deltas)
    vol = std(returns; dims=:obs)
    # TODO: price-impact ?
    return first(vol)  # vol is a 1-element NamedDimsArray hence first()
end

function volatility(volumes::AbstractVector, deltas::Sampleable{Multivariate})
    # Note, although `sqrtcov(deltas)` is in size `variable_length * node`,
    # `sqrtcov(deltas) * volumes` is a vector of length `variable_length`.
    # Taking the L2 norm of it, it becomes a scalar.
    # In short, although `sqrtcov(deltas)` is not unique, the `volatility`
    # function is calculating a unique scalar: `sqrt(volume' * cov(delta) * volume)`
    return norm(sqrtcov(deltas) * volumes, 2)
end
volatility(returns) = std(returns)
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
    volumes::AbstractVector, deltas::Union{Sampleable{Multivariate}, AbstractMatrix}, args...,
)
    mean_return = expected_return(volumes, deltas, args...)
    std_return = volatility(volumes, deltas)
    return mean_return / std_return
end

sharpe_ratio(returns) =  mean(returns) / std(returns)

obs_arrangement(::typeof(sharpe_ratio)) = MatrixColsOfObs()


"""
    median_return(returns, args...) -> Number
    median_return(volumes::AbstractVector, deltas::AbstractMatrix, args...) -> Number

Calculate the median return of a portfolio of `volumes` given a sample of price `deltas`.

## Arguments
- `returns`: an iterator of portfolio returns
- `volumes::AbstractVector`: The MWs volumes of the portfolio
- `deltas`::`AbstractMatrix`: The collection of prices deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).

We currently don't have a working version of this for Multivariate Distributions as there
are many definitions of `median` which aren't implemented by `Distributions`.
https://invenia.slack.com/archives/CMMAKP97H/p1567612804011200?thread_ts=1567543537.008300&cid=CMMAKP97H
"""
function median_return(volumes::AbstractVector, deltas::AbstractMatrix, args...)
    returns = _calculate_returns(volumes, deltas)
    pi = price_impact(volumes, args...)

    return median_return(returns) - pi
end

median_return(returns) = median(returns)

obs_arrangement(::typeof(median_return)) = MatrixColsOfObs()
