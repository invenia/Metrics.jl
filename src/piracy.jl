#=
    this file includes some functions we defined/extended for `Distributions.jl`.
    They are not used outside the `Metrics.jl` and should probably settle down
    in `Distributions.jl` properly in the long term.
=#
using StatsUtils
import Distributions: dof

"""
    GenericTDist

the TDist provided by Distributions.jl (https://github.com/JuliaStats/Distributions.jl/blob/997f2bbdc7d40982ec0a90e9aba7d0124b78bb52/src/univariate/continuous/tdist.jl)
doesn't have non-standard location and scale parameter, and hence define our own type.
"""
struct GenericTDist{T<:Real} <: ContinuousUnivariateDistribution
    df::T
    μ::T
    σ::T
    GenericTDist{T}(df::T, µ::T, σ::T) where {T<:Real} = new{T}(df, µ, σ)
end

"""
    dof

extract the degree of freedom parameter from a distribution
"""
dof(d::IndexedDistribution) = dof(parent(d))
dof(d::Union{GenericTDist, GenericMvTDist}) = d.df

StatsUtils.scale(d::GenericTDist) = d.σ
