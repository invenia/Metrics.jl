@testset "subsample.jl" begin

    seed!(1)

    # synthetic series for testing
    series = randn(1000, 10)

    @testset "block_subsample" begin
        block_series = collect(1:5)

        result = Metrics.block_subsample(block_series, 3)
        @test result == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

        @test_throws DomainError Metrics.block_subsample(block_series, 6)
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

        @testset "NaN result when non-finite weights are produced" begin
            β = 1.4
            x = 1:1:10

            # since all responses are the same the var of each one is 0 so weights are Infs
            y = map(_x -> repeat([_x], 50) .^ β, x)
            n = length(y)

            @test isnan(
                @test_logs(
                    (:warn, "10 non-finite weights arose in computing log-log slope."),
                    Metrics._compute_log_log_slope(x, y)
                )
            )
        end

        @testset "only works for Vector{Vector}" begin
            @test_throws MethodError Metrics._compute_log_log_slope(rand(10), rand(10))
            @test_throws MethodError Metrics._compute_log_log_slope(rand(10), rand(10, 4))
        end

    end

    @testset "estimate_convergence_rate" begin

        @testset "basic" begin
            result = mean(Metrics.estimate_convergence_rate.(eachcol(series), mean))
            # 0.5 is the theoretical value
            @test isapprox(result, 0.5, atol=0.01)
        end

        @testset "quantile range too short" begin
            @test_throws ArgumentError Metrics.estimate_convergence_rate(
                series, mean; quantmin=0.1, quantstep=0.2, quantmax=0.3
            )
        end

    end

    @testset "subsample_ci" begin

        @testset "basic" begin

            result = subsample_ci.(eachcol(series), mean; β=0.5)

            lower = mean(getfield.(result, :lower))
            upper = mean(getfield.(result, :upper))

            @test -0.1 < lower < 0.0
            @test 0.0 < upper < 0.1
            @test lower < mean(series) < upper

        end

        @testset "basic with block size" begin
            bs = Metrics.estimate_block_size(series, mean)

            # check that 3 arg form gives same result 2 arg form
            result_w_bs = subsample_ci(series, bs.block_size, mean; β=0.5)

            @test subsample_ci(series, mean; β=0.5) == result_w_bs
        end

        @testset "increasing alpha level contracts ci bounds" begin

            # this is expected behaviour since alpha is the level of the test we expect the
            # true value to lie in 1-alpha of the CIs
            r1 = subsample_ci(series, mean; α=0.1, β=0.5)
            r2 = subsample_ci(series, mean; α=0.2, β=0.5)

            @test r1.lower < r2.lower
            @test r1.upper > r2.upper

        end

        @testset "passing in kwargs for estimating block size" begin

            kwargs = (sizemin=20, sizemax=100, sizestep=1, blocksvol=3, β=0.123)

            ci_result = subsample_ci(series, mean; kwargs...)
            bs_result = Metrics.estimate_block_size(series, mean; kwargs...)

            @test bs_result.ci == ci_result
            @test ci_result == subsample_ci(series, bs_result.block_size, mean; β=0.123)

        end

    end


end
