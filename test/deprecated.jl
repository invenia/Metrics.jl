using Metrics: estimate_block_size, estimate_convergence_rate

@testset "deprecated.jl" begin
    series = rand(1000)

    @testset "estimate_convergence_rate" begin
        @test isequal(
            @test_deprecated(estimate_convergence_rate(series, mean)),
            estimate_convergence_rate(mean, series)
        )
    end

    @testset "estimate_block_size" begin
        @test isequal(
            @test_deprecated(estimate_block_size(series, mean)),
            estimate_block_size(mean, series)
        )
    end

    @testset "subsample_ci" begin
        @test isequal(
            @test_deprecated(subsample_ci(series, mean)),
            subsample_ci(mean, series)
        )

        @test isequal(
            @test_deprecated(subsample_ci(series, mean, 100)),
            subsample_ci(mean, series, 100)
        )
    end

end
