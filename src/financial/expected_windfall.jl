"""
    expected_windfall(returns; level::Real=0.05, per_mwh=false, volumes=[]) -> Number

Calculate the expected windfall `-𝔼[ r_p | r_p ≥ q_level(r_p) ]`, where `r_p` is
the portfolio return and `q_level(r_p)` is the upper quantile of the distribution
of `r_p` characterised by the `level`.

If an insufficient number of `returns` is provided to calculate the expected windfall
then this logs a warning and returns `missing`.

If `per_mwh=true`, returns the average return of the upper quantile divided by the average
volume of that quantile.

# Arguments
- `returns` (iterator): the portfolio of returns

# Keyword Arguments
- `level::Real`: level associated with the upper quantile of the returns distribution.
- `per_mwh`: compute expected windfall per MWh.
- `volumes`: volumes used in case `per_mwh=true`. Ignored otherwise.
"""
function expected_windfall(returns; level::Real=0.05, per_mwh=false, volumes=[])
    0 < level < 1 || throw(ArgumentError("level=$level is not between 0 and 1."))

    if per_mwh
        isempty(volumes) && throw(
            ArgumentError("The `volumes` keyword must be set when `per_mwh=true`.")
        )
        length(volumes) != length(returns) && throw(
            DimensionMismatch(
                """
                `returns` has length $(length(returns)) but `volumes` has
                length $(length(volumes)). Their lengths must be the same.
                """
            )
        )
    end

    returns = collect(returns) # required for iterators
    first_index = ceil(Int, (1 - level) * length(returns)) + 1
    last_index = length(returns)
    if first_index > length(returns)
        @warn(
            "Too few samples provided to calculate expected windfall for given level.",
             level,
             minimum_number_of_samples=ceil(Int, 1/level),
             number_of_samples=length(returns),
        )
        return missing
    end

    upper_quantile_inds = partialsortperm(returns, first_index:last_index)

    scale = per_mwh ? mean(@view volumes[upper_quantile_inds]) : 1.0
    return mean(@view returns[upper_quantile_inds]) / scale
end

ObservationDims.obs_arrangement(::typeof(expected_windfall)) = MatrixColsOfObs()
const ew = expected_windfall

"""
    expected_windfall(
        volumes::AbstractVector, deltas::AbstractArray, args...; kwargs...,
    ) -> Number

Calculate the sample expected windfall of the distribution of `returns` - with
[`price impact`](@ref price_impact) ``w'μ − w'Πw``, according to a certain `level`,
given a portfolio of `volumes` and a sample of price `deltas`.

# Arguments
- `volumes::AbstractVector`: the portfolio of volumes
- `deltas::AbstractArray`: collection of price deltas
- `args`: The [`price impact`](@ref price_impact) arguments (excluding `volumes`).

# Keyword Arguments
- `level::Real`: risk level associated with the lower quantile of the returns distribution
"""
function expected_windfall(
    volumes::AbstractVector, deltas::AbstractArray, args...; kwargs...,
)
    length(args) >= 3 && throw(MethodError(
        "Too many arguments. Please check `expected_windfall`'s docstring."
    ))
    returns = _calculate_returns(volumes, deltas)
    pi = price_impact(volumes, args...)

    if haskey(kwargs, :per_mwh) && kwargs[:per_mwh]
        vols = fill(sum(abs.(volumes)), length(returns))
        return expected_windfall(returns; volumes=vols, kwargs...) - pi
    end

    exp_windfall = expected_windfall(returns; kwargs...)

    return exp_windfall - pi
end
