using Distributions: Normal, MvNormal
using Metrics
using Metrics: IteratorOfObs, MatrixRowsOfObs, MatrixColsOfObs, arrange_obs
using NamedDims
using Random: seed!, shuffle!
using Test

@testset "Metrics.jl" begin
    include("evaluate.jl")
    include("utils.jl")
    include("regression.jl")
    include("financial.jl")
end
