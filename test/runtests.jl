using Distributions
using Metrics
using Metrics: IteratorOfObs, MatrixRowsOfObs, MatrixColsOfObs, organise_obs
using NamedDims
using Random: seed!, shuffle!
using Test

@testset "Metrics.jl" begin
    include("evaluate.jl")
    include("utils.jl")
    include("regression.jl")
    include("financial.jl")
end
