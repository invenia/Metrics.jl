using AxisArrays
using Distributions
using IndexedDistributions
using LinearAlgebra
using Metrics
using Metrics:
    IteratorOfObs,
    MatrixRowsOfObs,
    MatrixColsOfObs,
    ObsArrangement,
    obs_arrangement,
    organise_obs,
    price_impact,
    SingleObs
using NamedDims
using PSDMats
using Random
using Random: seed!
using Statistics
using StatsUtils: sqrtcov
using Test

import Statistics.mean

# Include test utilities
include("test_utils/properties.jl")
include("test_utils/stats.jl")

seed!(1)
@testset "Metrics.jl" begin
    include("utils.jl")
    # regression
    include("regression/gaussian_loglikelihood.jl")
    include("regression/picp.jl")
    include("regression/potential_payoff.jl")
    include("regression/simple.jl")
    # model
    include("model/kullback_leibler.jl")
    # financial
    include("financial/expected_shortfall.jl")
    include("financial/price_impact.jl")
    include("financial/simple.jl")
    # misc
    include("evaluate.jl")
    include("deprecated.jl")
    include("summary.jl")

end
