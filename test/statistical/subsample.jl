@testset "subsample.jl" begin

    seed!(1)

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

    @testset "_compute_log_log_slope" begin

        # the aim is to construct y = X ^ β synthetically and compute β
        @testset "basic" begin
            β = 1.4
            x = 1:1:10
            y = map(_x -> repeat([_x], 50) .^ β + rand(50), x)

            @test isapprox(Metrics._compute_log_log_slope(x, y), β, atol=0.1)
        end

        @testset "non-finite weights" begin
            β = 1.4
            x = 1:1:10
            y = map(_x -> repeat([_x], 50) .^ β, x)

            @test isnan(
                @test_logs (:warn, "Not all weights are finite.") Metrics._compute_log_log_slope(x, y)
            )
        end

        @testset "only works for Vector{Vector}" begin
            @test_throws MethodError Metrics._compute_log_log_slope(rand(10), rand(10))
            @test_throws MethodError Metrics._compute_log_log_slope(rand(10), rand(10, 4))
        end

    end
end
