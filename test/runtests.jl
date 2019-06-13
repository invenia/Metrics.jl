using Metrics
using Test

using Distributions: Normal, MvNormal

@testset "Metrics.jl" begin
    include("utils.jl")
    include("regression.jl")
end
