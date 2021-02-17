#=
    this file includes some functions we defined/extended for `Distributions.jl`.
    They are not used outside the `Metrics.jl` and should probably settle down
    in `Distributions.jl` properly in the long term.
=#
using Distributions: @check_args
import Distributions: dof

"""
    GenericTDist{T<:Real}(df::T, μ::T, σ::T)

Univariate T distribution with location and scale parameters. The reason we have to define
this type is because that the TDist provided by Distributions.jl doesn't have non-standard
location and scale parameter.

# Arguments
- `df::Real`: the degree of freedom
- `μ::Real`: the location parameter
- `σ::Real`: the scale parameter
"""
struct GenericTDist{T<:Real} <: ContinuousUnivariateDistribution
    df::T
    μ::T
    σ::T
    GenericTDist{T}(df::T, µ::T, σ::T) where {T<:Real} = new{T}(df, µ, σ)
end

function GenericTDist(df::T, μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(GenericTDist, (σ >= zero(σ)) && (df > zero(df)))
    return GenericTDist{T}(df, μ, σ)
end

dof(d::IndexedDistribution) = dof(parent(d))
dof(d::Union{GenericTDist, GenericMvTDist}) = d.df

Distributions.mean(d::GenericTDist) = d.df>1 ? d.μ : NaN
Distributions.scale(d::GenericTDist) = d.σ
