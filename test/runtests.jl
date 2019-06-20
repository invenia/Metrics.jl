using Metrics
using Test

using Random: seed!
using Distributions: Normal, MvNormal

@testset "Metrics.jl" begin
    include("utils.jl")
    include("regression.jl")
end
