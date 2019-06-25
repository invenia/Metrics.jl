using Metrics
using Test

using Random: seed!, shuffle!
using Distributions: Normal, MvNormal

@testset "Metrics.jl" begin
    include("utils.jl")
    include("regression.jl")
    include("financial.jl")
end
