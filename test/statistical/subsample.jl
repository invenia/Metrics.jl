@testset "subsample.jl" begin

    @testset "main" begin
        β = mean([estimate_convergence_rate(randn(1000), mean) for _ in 1:200])
        @test 0.48 <= β <= 0.52 # theoretical value is 0.50.
        cis = [subsample_ci(randn(1000), mean; β=0.5) for _ in 1:200]
        low = mean([ci[1] for ci in cis])
        up = mean([ci[2] for ci in cis])
        @test -0.1 < low < 0.0
        @test 0.0 < up < 0.1
    end

    @testset "_compute_quantile_differences" begin

        @testset "basic" begin
            samples = [1:5, 1:5]
            qmin = 0.2
            qstep = 0.2
            qmax = 0.8

            expected = [
                [0.8, 0.7999999999999998, 0.8000000000000003],
                [0.8, 0.7999999999999998, 0.8000000000000003]
            ]

            result = Metrics._compute_quantile_differences(samples, qmin, qstep, qmax)

            @test result == expected
        end

        @testset "errors when quant range is too short" begin
            samples = [1:5, 1:5]
            @test_throws ArgumentError Metrics._compute_quantile_differences(samples, 0.1, 0.2, 0.3)
        end
    end
end
