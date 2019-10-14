using Distributions
using IndexedDistributions
using LinearAlgebra
using Metrics
using Metrics:
    IteratorOfObs,
    MatrixRowsOfObs,
    MatrixColsOfObs,
    SingleObs,
    obs_arrangement,
    organise_obs,
    price_impact
using NamedDims
using PSDMats
using Random
using Random: seed!
using Statistics
using StatsUtils: sqrtcov
using Test

import Statistics.mean

include("test_utils.jl")

# Include test_utils
include("test_utils.jl")

seed!(1)
@testset "Metrics.jl" begin
    include("evaluate.jl")
    include("utils.jl")
    include("regression.jl")
    include("regression_picp.jl")
    include("kullback_leibler.jl")
    include("financial.jl")
    include("price_impact.jl")
    include("summaries.jl")
    include("deprecated.jl")
end
