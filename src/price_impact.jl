using Metrics: @_dimcheck

# Note: these functions should eventually be moved to the PriceImpacts.jl package

"""
    price_impact(volumes::AbstractVector, Pi::AbstractMatrix) -> Number

Calculate price impact ``w'Πw`` on the expected return with a price impact matrix `Pi` and
`volumes` vector.

NOTE: this is a linear model of price impact. For our MWs `w` and observed LMP deltas `o`,
the counterfactual LMP deltas are modelled as ``c = o - Πw``, so ``\$ = w'c = w'o - w'Πw``.
"""
function price_impact(volumes::AbstractVector, Pi::AbstractMatrix)::Number
    return volumes' * Pi * volumes  # w'Πw
end


"""
    price_impact(volumes::AbstractVector, supply_pi, demand_pi) -> Number
    price_impact(supply::AbstractVector, demand::AbstractVector, supply_pi, demand_pi) -> Number

Calculate price impact ``w'Πw`` on the expected return on a vector of `volumes` or the
constituent `supply` and `demand` vectors.

`supply_pi` and `demand_pi` are the diagonal elements of the price impact matrix which
assumes the price impact of each node is independent.
"""
function price_impact(volumes::AbstractVector, supply_pi, demand_pi)::Number
    supply, demand = split_volume(volumes)
    return price_impact(supply, demand, supply_pi, demand_pi)
end

function price_impact(
    supply::AbstractVector, demand::AbstractVector, supply_pi, demand_pi,
)::Number
    # Local, linear price impact: `Π` is a diagonal matrix
    # w'Πw = a'Sa + b'Db, where w = a + b; Π = S + D
    @_dimcheck length(supply) == length(demand)
    (iszero(supply_pi) && iszero(demand_pi)) && return 0.0

    N = length(supply)
    s_pi = isa(supply_pi, Number) ? fill(supply_pi, N) : supply_pi
    d_pi = isa(demand_pi, Number) ? fill(demand_pi, N) : demand_pi
    return dot(s_pi, supply.^2) + dot(d_pi, demand.^2)
end

# if no PI parameters input then default to 0
price_impact(volumes::AbstractVector) = 0


"""
    split_volume(volumes::AbstractVector) -> Tuple{<:AbstractVector}

Split combined mega-watts (`volumes`) into two vectors the same length as the input, with one
containing supply (positive values) and the other demand (negative values) mega-watts.
"""
function split_volume(volumes::AbstractVector)
    supply = max.(volumes, 0.0)
    demand = min.(volumes, 0.0)
    return supply, demand
end
