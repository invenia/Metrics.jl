@testset "subsample.jl" begin
    β = mean([estimate_convergence_rate(randn(1000), mean) for _ in 1:200])
    @test 0.48 <= β <= 0.52 # theoretical value is 0.50.
    cis = [subsample_ci(randn(1000), mean; β=0.5) for _ in 1:200]
    low = mean([ci[1] for ci in cis])
    up = mean([ci[2] for ci in cis])
    @test -0.1 < low < 0.0
    @test 0.0 < up < 0.1
end
