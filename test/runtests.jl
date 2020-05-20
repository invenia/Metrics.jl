using AxisArrays
using Distributions
using FixedPointDecimals
using IndexedDistributions
using LinearAlgebra
using Metrics
using Metrics: price_impact
using ObservationDims
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
    # financial
    include("financial/expected_shortfall.jl")
    include("financial/expected_windfall.jl")
    include("financial/price_impact.jl")
    include("financial/simple.jl")
    # statistical
    include("statistical/bky.jl")
    include("statistical/diebold_mariano.jl")
    include("statistical/kullback_leibler.jl")
    include("statistical/subsample.jl")
    # misc
    include("evaluate.jl")
    include("summary.jl")
    include("deprecated.jl")
end
