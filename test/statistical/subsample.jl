@testset "subsample.jl" begin

    seed!(1)

    # synthetic series for testing
    series = randn(1000, 10)
    series_a = series[:, 1]
    series_b = series[:, 2]


    @testset "block_subsample" begin
        block_series = collect(1:5)

        result = Metrics.block_subsample(block_series, 3)
        @test result == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

        result = Metrics.block_subsample(block_series, length(block_series))
        @test length(result) == 1

        result = Metrics.block_subsample(block_series, 3, circular=true)
        @test result == [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 1], [5, 1, 2]]

        result = Metrics.block_subsample(block_series, length(block_series), circular=true)
        @test length(result) == length(block_series)

        result = Metrics.block_subsample(block_series, 1)
        result_c = Metrics.block_subsample(block_series, 1, circular=true)
        @test length(result) == length(block_series)
        @test result == result_c

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
            result = mean(Metrics.estimate_convergence_rate.(mean, eachcol(series)))
            # 0.5 is the theoretical value
            @test isapprox(result, 0.5, atol=0.01)
        end

        @testset "quantile range too short" begin
            @test_throws ArgumentError Metrics.estimate_convergence_rate(
                mean, series; quantmin=0.1, quantstep=0.2, quantmax=0.3
            )
        end

    end

    # These are the old default arguments for estimating block sizes. We will use them for
    # the tests because: 1. it preserves the old numerical comparisons; 2. it leads to much
    # faster tests, as the current defaults scale with the data set size, and we use a very
    # large one in these tests.
    old_block_args = (sizemin=50, sizemax=300)

    @testset "adaptive_block_size" begin
        # 1001 > length(series_a) = 1000
        @test_throws DomainError Metrics.adaptive_block_size(mean, series_a, sizemin=998, sizemax=1001)

        @test_throws DimensionMismatch Metrics.adaptive_block_size(mean, randn(5), randn(4))
    end

    @testset "Function distance" begin
        f1(x) = x
        f2(x) = 2x
        loc = collect(-1:4)
        @test Metrics.l2_distance(f1, f1, loc) == 0
        @test Metrics.l2_distance(f1, f2, loc) == 1 + 1 + 4 + 9 + 16
    end

    @testset "subsample_ci and subsample_difference_ci" begin

        @testset "basic" begin
            result = subsample_ci.(mean, eachcol(series); old_block_args...)

            lower = mean(first, result)
            upper = mean(last, result)

            @test -0.1 < lower < 0.0
            @test 0.0 < upper < 0.1
            @test lower < mean(series) < upper

            result = map(
                x -> subsample_difference_ci(mean, x[1], x[2]; sizemin=4, sizemax=80),
                [(randn(1000), randn(1000)) for _ in 1:50],
            )

            lower = mean(first, result)
            upper = mean(last, result)

            @test -0.1 < lower < 0.0
            @test 0.0 < upper < 0.1
        end

        @testset "basic with block size" begin
            bs = Metrics.adaptive_block_size(mean, series; old_block_args...)

            # check that 3 arg form gives same result 2 arg form
            result_w_bs = subsample_ci(mean, series, bs)
            @test subsample_ci(mean, series; β=0.5, old_block_args...) == result_w_bs

            # test that studentisation affects the result
            @test subsample_ci(mean, series; studentize=true, β=0.5, old_block_args...) !=
                subsample_ci(mean, series; studentize=false, β=0.5, old_block_args...)

            # test that circular affects the result
            @test subsample_ci(mean, series; circular=true, β=0.5, old_block_args...) !=
                subsample_ci(mean, series; circular=false, β=0.5, old_block_args...)

            bs = Metrics.adaptive_block_size(mean, series_a, series_b; old_block_args...)

            # check that 3 arg form gives same result 2 arg form
            result_w_bs = subsample_difference_ci(mean, series_a, series_b, bs)
            result_no_bs = subsample_difference_ci(
                mean, series_a, series_b; β=0.5, old_block_args...
            )
            @test result_no_bs == result_w_bs

            # test that studentisation affects the result
            @test subsample_difference_ci(mean, series_a, series_b; studentize=true, β=0.5, old_block_args...) !=
                subsample_difference_ci(mean, series_a, series_b; studentize=false, β=0.5, old_block_args...)

            @test subsample_pct_difference_ci(mean, series_a, series_b; studentize=true, β=0.5, old_block_args...) !=
                subsample_pct_difference_ci(mean, series_a, series_b; studentize=false, β=0.5, old_block_args...)

            # test that circular affects the result
            @test subsample_difference_ci(mean, series_a, series_b; circular=true, β=0.5, old_block_args...) !=
                subsample_difference_ci(mean, series_a, series_b; circular=false, β=0.5, old_block_args...)

            @test subsample_pct_difference_ci(mean, series_a, series_b; circular=true, β=0.5, old_block_args...) !=
                subsample_pct_difference_ci(mean, series_a, series_b; circular=false, β=0.5, old_block_args...)
        end

        @testset "linearity of differences in mean" begin
            diff_ci = subsample_difference_ci(mean, series_a, series_b; β=0.5, old_block_args...)
            ci_diff = subsample_ci(mean, series_a .- series_b; β=0.5, old_block_args...)
            @test (first(diff_ci) ≈ first(ci_diff))
            @test (last(diff_ci) ≈ last(ci_diff))
        end

        @testset "antisymmetry of differences" begin
            for metric in [mean, median, es, mean_over_es, ew, ew_over_es]
                diff_1 = subsample_difference_ci(metric, series_a, series_b; β=0.5, old_block_args...)
                diff_2 = subsample_difference_ci(metric, series_b, series_a; β=0.5, old_block_args...)
                @test (first(diff_1) ≈ -last(diff_2))
                @test (last(diff_1) ≈ -first(diff_2))
            end
        end

        @testset "increasing alpha level contracts ci bounds" begin

            # this is expected behaviour since alpha is the level of the test we expect the
            # true value to lie in 1-alpha of the CIs
            r1 = subsample_ci(mean, series; α=0.1, β=0.5, old_block_args...)
            r2 = subsample_ci(mean, series; α=0.2, β=0.5, old_block_args...)

            @test first(r1) < first(r2)
            @test last(r1) > last(r2)

        end

        @testset "passing kwargs" begin

            block_kwargs = (sizemin=20, sizemax=100, sizestep=2, numpoints=50)
            conv_kwargs = (quantmin=0.2, quantstep=0.01, quantmax=0.7, expmax=0.4, expstep=0.1, expmin=0.10)

            @testset "estimate block size" begin
                ci_result = subsample_ci(mean, series; β=0.123, block_kwargs...)
                bs_result = Metrics.adaptive_block_size(mean, series; block_kwargs...)
                @test ci_result == subsample_ci(mean, series, bs_result; β=0.123)

                ci_result = subsample_difference_ci(mean, series_a, series_b; β=0.123, block_kwargs...)
                bs_result = Metrics.adaptive_block_size(mean, series_a, series_b; block_kwargs...)
                @test ci_result == subsample_difference_ci(mean, series_a, series_b, bs_result; β=0.123)
            end

            @testset "estimate convergence rate" begin
                ci_result = subsample_ci(mean, series; β=nothing, old_block_args..., conv_kwargs...)
                beta = Metrics.estimate_convergence_rate(mean, series; conv_kwargs...)
                @test ci_result == subsample_ci(mean, series; β=beta, old_block_args...)

                ci_result = subsample_difference_ci(
                    mean, series_a, series_b;
                    β=nothing, old_block_args..., conv_kwargs...
                )
                # Pair observations
                paired_series = collect(zip(series_a, series_b))
                # Define metric from R² to R
                diff_mean = x -> mean(getfield.(x, 1)) - mean(getfield.(x, 2))
                beta = Metrics.estimate_convergence_rate(diff_mean, paired_series; conv_kwargs...)
                @test ci_result == subsample_difference_ci(mean, series_a, series_b; β=beta, old_block_args...)

                ci_result = subsample_pct_difference_ci(
                    mean, series_a, series_b;
                    β=nothing, old_block_args..., conv_kwargs...
                )
                pct_diff_mean = x -> 100 * (mean(getfield.(x, 1)) - mean(getfield.(x, 2))) / (abs(mean(last.(x))) + 1e-5)
                beta = Metrics.estimate_convergence_rate(pct_diff_mean, paired_series; conv_kwargs...)
                @test ci_result == subsample_pct_difference_ci(mean, series_a, series_b; β=beta, old_block_args...)

                # Warnings
                warning = r"extra_kwarg"
                @test_logs (:warn, warning) subsample_difference_ci(
                    mean, series_a, series_b; # Without block_size
                    β=beta, extra_kwarg=20, old_block_args...
                )

                @test_logs (:warn, warning) subsample_difference_ci(
                    mean, series_a, series_b, 12; # With block_size
                    β=beta, extra_kwarg=20, old_block_args...
                )

                @test_logs (:warn, warning) subsample_ci(
                    mean, series; # Without block_size
                    β=beta, extra_kwarg=20, old_block_args...
                )

                @test_logs (:warn, warning) subsample_ci(
                    mean, series, 12; # With block_size
                    β=beta, extra_kwarg=20, old_block_args...
                )
            end

            @testset "estimate block size" begin
                ci_result = subsample_ci(mean, series; β=nothing, block_kwargs..., conv_kwargs...)
                bs_result = Metrics.adaptive_block_size(mean, series; block_kwargs...)
                @test ci_result == subsample_ci(mean, series, bs_result; β=nothing, conv_kwargs...)

                ci_result = subsample_difference_ci(
                    mean, series_a, series_b;
                    β=nothing, block_kwargs..., conv_kwargs...
                )
                bs_result = Metrics.adaptive_block_size(mean, series_a, series_b; block_kwargs...)
                # Pair observations
                paired_series = collect(zip(series_a, series_b))
                # Define metric from R² to R
                diff_mean = x -> mean(getfield.(x, 1)) - mean(getfield.(x, 2))
                @test ci_result == subsample_difference_ci(
                    mean, series_a, series_b, bs_result; β=nothing, conv_kwargs...
                )
            end

            @testset "estimate convergence rate" begin
                ci_result = subsample_ci(mean, series; β=nothing, block_kwargs..., conv_kwargs...)
                beta = Metrics.estimate_convergence_rate(mean, series; conv_kwargs...)
                @test ci_result == subsample_ci(mean, series; β=beta, block_kwargs...)

                ci_result = subsample_difference_ci(
                    mean, series_a, series_b;
                    β=nothing, block_kwargs..., conv_kwargs...
                )
                # Pair observations
                paired_series = collect(zip(series_a, series_b))
                # Define metric from R² to R
                diff_mean = x -> mean(getfield.(x, 1)) - mean(getfield.(x, 2))
                beta = Metrics.estimate_convergence_rate(diff_mean, paired_series; conv_kwargs...)
                @test ci_result == subsample_difference_ci(
                    mean, series_a, series_b; β=beta, block_kwargs...
                )

                ci_result = subsample_pct_difference_ci(
                    mean, series_a, series_b;
                    β=nothing, block_kwargs..., conv_kwargs...
                )
                # Pair observations
                paired_series = collect(zip(series_a, series_b))
                # Define metric from R² to R
                pct_diff_mean = x -> 100 * (mean(getfield.(x, 1)) - mean(getfield.(x, 2))) / (abs(mean(last.(x))) + 1e-5)
                beta = Metrics.estimate_convergence_rate(pct_diff_mean, paired_series; conv_kwargs...)
                @test ci_result == subsample_pct_difference_ci(
                    mean, series_a, series_b; β=beta, block_kwargs...
                )
            end

            @testset "default β" begin
                series = rand(400)

                for metric in [
                    mean,
                    median,
                    mean_over_es,
                    median_over_es,
                    expected_shortfall,
                    expected_windfall,
                    ew_over_es,
                ]
                    # Choosing a large sizemin so that ES and EW have sufficient samples to be computed.
                    # Choosing a small range of sizes to save time.
                    ci_result = subsample_ci(metric, series, sizemin=40, sizemax=100)
                    @test ci_result == subsample_ci(
                        metric, series, β=0.5, sizemin=40, sizemax=100
                    )
                    ci_result = subsample_ci(metric, series, 100)
                    @test ci_result == subsample_ci(metric, series, 100, β=0.5)
                end
            end

            @testset "default sizemin" begin
                sseries1 = rand(200)
                sseries2 = rand(200)

                for metric in [
                    mean_over_es,
                    median_over_es,
                    expected_shortfall,
                    expected_windfall,
                    ew_over_es,
                ]
                    ci_result = subsample_ci(metric, sseries1, sizemin=40, sizemax=100)
                    @test ci_result == subsample_ci(metric, sseries1, sizemax=100)

                    ci_result = subsample_difference_ci(metric, sseries1, sseries2, sizemin=40, sizemax=100)
                    @test ci_result == subsample_difference_ci(metric, sseries1, sseries2, sizemax=100)

                    ci_result = subsample_pct_difference_ci(metric, sseries1, sseries2, sizemin=40, sizemax=100)
                    @test ci_result == subsample_pct_difference_ci(metric, sseries1, sseries2, sizemax=100)
                end

                for metric in [
                    mean,
                    median,
                    x -> mean(x) + median(x), # just a random lambda
                ]
                    ci_result = subsample_ci(metric, sseries1, sizemin=4, sizemax=100)
                    @test ci_result == subsample_ci(metric, sseries1, sizemax=100)

                    ci_result = subsample_difference_ci(metric, sseries1, sseries2, sizemin=4, sizemax=100)
                    @test ci_result == subsample_difference_ci(metric, sseries1, sseries2, sizemax=100)

                    ci_result = subsample_pct_difference_ci(metric, sseries1, sseries2, sizemin=4, sizemax=100)
                    @test ci_result == subsample_pct_difference_ci(metric, sseries1, sseries2, sizemax=100)
                end
            end

        end

        @testset "do-block" begin

            result = subsample_ci(series; old_block_args...) do s
                mean(s)
            end

            @test result == subsample_ci(mean, series; β=nothing, old_block_args...)

        end

    end
end
