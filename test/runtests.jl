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
using Distributions

seed!(1)
@testset "Metrics.jl" begin
    include("evaluate.jl")
    include("utils.jl")
    include("regression.jl")
    include("regression_picp.jl")
    include("divergence.jl")
    include("financial.jl")
    include("price_impact.jl")
end
