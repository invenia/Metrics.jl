using Distributions
using LinearAlgebra
using Metrics
using Metrics:
    IteratorOfObs,
    MatrixRowsOfObs,
    MatrixColsOfObs,
    organise_obs,
    price_impact
using NamedDims
using Random
using Random: seed!
using Statistics
using StatsUtils: sqrtcov
using Test

@testset "Metrics.jl" begin
    include("evaluate.jl")
    include("utils.jl")
    include("regression.jl")
    include("price_impact.jl")
    include("financial.jl")
end
