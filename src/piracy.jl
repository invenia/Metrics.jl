#=
    this file includes some functions we defined/extended for `Distributions.jl`.
    They are not used outside the `Metrics.jl` and should probably settle down
    in `Distributions.jl` properly in the long term.
=#
using Distributions: @check_args
import Distributions: dof

"""
    GenericTDist

the TDist provided by Distributions.jl doesn't have non-standard location and scale
parameter, and hence define our own type.
"""
struct GenericTDist{T<:Real} <: ContinuousUnivariateDistribution
    df::T
    μ::T
    σ::T
    GenericTDist{T}(df::T, µ::T, σ::T) where {T<:Real} = new{T}(df, µ, σ)
end

function GenericTDist(df::T, μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(GenericTDist, σ >= zero(σ))
    return GenericTDist{T}(df, μ, σ)
end

"""
    dof

extract the degree of freedom parameter from a distribution
"""
dof(d::IndexedDistribution) = dof(parent(d))
dof(d::Union{GenericTDist, GenericMvTDist}) = d.df

Distributions.mean(d::GenericTDist) = d.μ
Distributions.scale(d::GenericTDist) = d.σ
