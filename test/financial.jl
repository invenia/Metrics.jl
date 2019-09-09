using Metrics: split_volume

# helper function to generate MvNormals
function generate_mvnormal(mean, size::Integer)
    X = rand(size, size)
    rand_cov = X' * X + 0.01I
    return MvNormal(mean, Symmetric(rand_cov))
end
generate_mvnormal(size::Integer) = generate_mvnormal(rand(size), size)

@testset "financial.jl" begin

    @testset "expected return" begin

        @testset "basic properties" begin

            # simple use of base function
            volumes = [0, 1, -10, 100]
            deltas = [0.1, 0.2, 0.3, 0.4]
            expected = 37.2
            @test expected_return(volumes, deltas) == expected
            @test evaluate(expected_return, volumes, deltas) == expected

            # negating both input should make no difference
            @test isequal(
                expected_return(volumes, deltas),
                expected_return(-volumes, -deltas)
            )
            # negating one input should flip the sign on the result
            @test isequal(
                expected_return(volumes, -deltas),
                expected_return(-volumes, deltas)
            )
            @test isequal(
                expected_return(volumes, deltas),
                -expected_return(-volumes, deltas)
            )

            # with price impact
            nonzero_pi = (supply_pi=fill(0.1, 4), demand_pi=fill(0.1, 4))
            supply, demand = split_volume(volumes)
            pi = dot(nonzero_pi.supply_pi, supply.^2) + dot(nonzero_pi.demand_pi, demand.^2)

            # update previous expected result
            expected -= pi
            @test expected_return(volumes, deltas, nonzero_pi...) == expected
            @test expected < expected_return(volumes, deltas)

        end


        # using dense cov matrix
        num_nodes = 20
        volumes = rand(Uniform(-50,50), num_nodes)
        mean_deltas = rand(num_nodes)
        dense_dist = generate_mvnormal(mean_deltas, num_nodes)
        nonzero_pi = (supply_pi=fill(0.1, num_nodes), demand_pi=fill(0.1, num_nodes))

        @testset "with distribution" begin
            expected = dot(volumes, mean_deltas)
            @test expected_return(volumes, dense_dist) ≈ expected
            @test evaluate(expected_return, volumes, dense_dist) ≈ expected

            # with price impact
            @test expected_return(volumes, dense_dist, nonzero_pi...) < expected
            @test evaluate(expected_return, volumes, dense_dist, nonzero_pi...) < expected
        end

        @testset "with samples" begin
            # using sample deltas
            samples = rand(dense_dist, 10)
            expected = dot(volumes, mean(samples, dims=2))
            @test expected_return(volumes, samples) ≈ expected
            @test evaluate(expected_return, volumes, samples; obsdim=2) ≈ expected

            @test expected_return(volumes, samples, nonzero_pi...) < expected
            @test isless(
                evaluate(expected_return, volumes, samples, nonzero_pi...; obsdim=2),
                expected,
            )
        end

     end

    @testset "volatility" begin

        # using diagonal cov matrix
        diag_sqrtcov = Diagonal([5.0, 6.0 ,7.0])
        diag_dist = MvNormal(rand(3), diag_sqrtcov' * diag_sqrtcov)
        @test isequal(
            volatility([0.1, 1.0, -10.0], diag_dist),
            sqrt(0.5^2 + 6^2 + 70^2)
        )

        # using dense cov matrix
        volumes = rand(Uniform(-50,50), 10)
        dense_dist = generate_mvnormal(10)

        @testset "with distribution" begin
            expected = norm(sqrtcov(dense_dist) * volumes, 2)
            @test volatility(volumes, dense_dist) ≈ expected
            @test evaluate(volatility, volumes, dense_dist) ≈ expected
        end

        @testset "with samples" begin
            samples = rand(dense_dist, 5)
            expected = std(samples' * volumes)
            @test volatility(volumes, samples) ≈ expected
            @test evaluate(volatility, volumes, samples; obsdim=2) ≈ expected
        end

    end

    @testset "sharpe ratio" begin

        # using diag cov matrix
        vol = [0.1, 0.2, -0.3]
        diag_sqrtcov = Diagonal([5.0, 6.0 ,7.0])
        diag_dist = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)
        exp_return = (0.2 + -3.0)
        exp_vol = sqrt(0.5^2 + 1.2^2 + 2.1^2)
        @test sharpe_ratio(vol, diag_dist) ≈ exp_return / exp_vol

        # using dense cov matrix
        volumes = rand(Uniform(-50,50), 10)
        dense_dist = generate_mvnormal(10)
        nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))

        @testset "with distribution" begin
            exp_return = expected_return(volumes, dense_dist)
            exp_vol = volatility(volumes, dense_dist)

            @test sharpe_ratio(volumes, dense_dist) == exp_return / exp_vol
            @test evaluate(sharpe_ratio, volumes, dense_dist) == exp_return / exp_vol

            # with price impact
            @test sharpe_ratio(volumes, dense_dist, nonzero_pi...) < exp_return / exp_vol
            @test isless(
                evaluate(sharpe_ratio, volumes, dense_dist, nonzero_pi...),
                exp_return / exp_vol,
            )
        end

        @testset "with samples" begin
            samples = rand(dense_dist, 5)

            exp_return = expected_return(volumes, samples)
            exp_vol = volatility(volumes, samples)

            @test sharpe_ratio(volumes, samples) == exp_return / exp_vol
            @test evaluate(sharpe_ratio, volumes, samples; obsdim=2) ≈ exp_return / exp_vol

            # with price impact sharpe ratio should decrease
            @test sharpe_ratio(volumes, samples, nonzero_pi...) < exp_return / exp_vol
            @test isless(
                evaluate(sharpe_ratio, volumes, samples, nonzero_pi...; obsdim=2),
                exp_return / exp_vol,
            )
        end
    end

    @testset "median_return" begin
        # Basic Usage
        volumes = collect(1:10)
        deltas = Matrix(I, (10, 10))
        expected = 5.5

        @test median_return(volumes, deltas) == expected
        @test evaluate(median_return, volumes, deltas) == expected

        # With Price Impact median_return should be lower
        nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))
        pi_expected = -33

        @test median_return(volumes, deltas, nonzero_pi...) == pi_expected
        @test evaluate(median_return, volumes, deltas, nonzero_pi...) == pi_expected
        @test pi_expected < expected
    end

    @testset "evano" begin

        @testset "simple evano" begin
            # Basic usage
            returns = collect(1:100)

            # Default risk_level is 0.05
            expected = -16.833333333333332
            @test evano(returns) == expected

            # Order shouldn't matter
            shuffle!(returns)
            @test evano(returns) == expected

            # Testing a different risk_level
            risk_level = 0.25
            expected = -3.8846153846153846
            @test evano(returns; risk_level=risk_level) == expected
            @test evaluate(evano, returns; risk_level=risk_level) == expected
        end

        @testset "sample evano" begin
            # Using samples
            volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
            samples = Matrix(I, (10, 10))
            expected = -0.08333333333333333
            @test evano(volumes, samples; risk_level=0.5) == expected
            @test evaluate(evano, volumes, samples; risk_level=0.5) == expected

            # using diagonal matrix of samples - requires AbstractArray
            sample_deltas = Diagonal(1:10)
            expected = -0.03409090909090909
            @test evano(volumes, sample_deltas; risk_level=0.5) == expected

            # with price impact evano should decrease since expected shortfall increases
            nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))
            @test evano(volumes, sample_deltas, nonzero_pi...; risk_level=0.5) < expected
            @test isless(
                evaluate(
                    evano, volumes, sample_deltas, nonzero_pi...; risk_level=0.5, obsdim=2
                ),
                expected,
            )

            # generate samples from distribution of deltas
            seed!(1234)
            volumes = repeat([1, -2, 3, -4, 5, -6, 7, -8, 9, -10], 2)
            samples = rand(MvNormal(ones(20)), 50)

            expected = 0.08900853620347395
            @test evano(volumes, samples) ≈ expected
            @test evaluate(evano, volumes, samples; obsdim=2) ≈ expected

            # too few samples
            @test_throws ArgumentError evano(volumes, samples; risk_level=0.01)

            # single sample should not work given `risk_level=1` but not otherwise.
            @test_throws MethodError evano([-5], [10]; risk_level=0.99)
        end

        @testset "analytic evano" begin
            # We currently don't have a working version of this for Multivariate
            # Distributions as there are many definitions of `median` which aren't
            # implemented by `Distributions`.
            # https://invenia.slack.com/archives/CMMAKP97H/p1567612804011200?thread_ts=1567543537.008300&cid=CMMAKP97H
            # More info: https://www.r-bloggers.com/multivariate-medians/
            seed!(1)
            volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
            dense_dist = generate_mvnormal(10)
            nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))

            # Using an MvNormal won't work since we don't have a good way of calculating the
            # median yet
            @test_throws MethodError evano(volumes, dense_dist; risk_level=0.5)
            @test_throws MethodError evaluate(evano, volumes, dense_dist; risk_level=0.5)

            # with price impact, this should still fail
            @test_throws MethodError evano(
                volumes, dense_dist, nonzero_pi...; risk_level=0.01
            )
            @test_throws MethodError evaluate(
                evano, volumes, dense_dist, nonzero_pi...; risk_level=0.01
            )
        end
    end

    @testset "expected shortfall" begin

        @testset "simple ES" begin

            # basic usage
            returns = collect(1:100)

            # default risk_level is 0.05
            @test expected_shortfall(returns) == -3

            # order shouldn't matter
            shuffle!(returns)
            @test expected_shortfall(returns) == -3

            # testing a different risk_level
            risk_level = 0.25
            @test expected_shortfall(returns; risk_level=risk_level) == -13
            @test evaluate(expected_shortfall, returns; risk_level=risk_level) == -13

            # replacing larger values shouldn't matter
            returns_2 = sort(returns)
            returns_2[6:end] .= 1000
            shuffle!(returns_2)
            @test expected_shortfall(returns) == expected_shortfall(returns_2)
        end

        @testset "sample ES" begin

            # using samples
            volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
            samples = Matrix(I, (10, 10))
            expected = -mean([-2, -4, -6, -8, -10])  # = 6.0
            @test expected_shortfall(volumes, samples; risk_level=0.5) == expected
            @test evaluate(expected_shortfall, volumes, samples; risk_level=0.5) == expected

            # using diagonal matrix of samples - requires AbstractArray
            sample_deltas = Diagonal(1:10)
            expected =  -mean([-2*2, -4*4, -6*6, -8*8, -10*10])  # = 44.0
            @test expected_shortfall(volumes, sample_deltas; risk_level=0.5) == expected

            # generate samples from distribution of deltas
            seed!(1234)
            volumes = repeat([1, -2, 3, -4, 5, -6, 7, -8, 9, -10], 2)
            samples = rand(MvNormal(ones(20)), 50)
            nonzero_pi = (supply_pi=fill(0.1, 20), demand_pi=fill(0.1, 20))

            expected = 48.304727988173816
            @test expected_shortfall(volumes, samples) ≈ expected
            @test evaluate(expected_shortfall, volumes, samples; obsdim=2) ≈ expected

            # with price impact ES should increase (due to sign)
            @test expected_shortfall(volumes, samples, nonzero_pi...) > expected
            @test isless(
                expected,
                evaluate(expected_shortfall, volumes, samples, nonzero_pi...; obsdim=2),
            )

            # too few samples
            @test_throws ArgumentError expected_shortfall(volumes, samples; risk_level=0.01)

            # single sample should not work given `risk_level=1` but not otherwise.
            @test_throws MethodError expected_shortfall([-5], [10]; risk_level=0.99)

        end

        @testset "analytic ES" begin

            seed!(1)
            volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
            dense_dist = generate_mvnormal(10)
            nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))

            # basic usage
            expected = 26.995589121396023
            @test expected_shortfall(volumes, dense_dist; risk_level=0.5) ≈ expected

            expected = 73.06492436615295
            @test expected_shortfall(volumes, dense_dist; risk_level=0.01) ≈ expected
            @test evaluate(expected_shortfall, volumes, dense_dist; risk_level=0.01) ≈ expected

            # with price impact ES should increase (due to sign)
            @test isless(
                expected,
                expected_shortfall(volumes, dense_dist, nonzero_pi...; risk_level=0.01),
            )
            @test isless(
                expected,
                evaluate(expected_shortfall, volumes, dense_dist, nonzero_pi...; risk_level=0.01),
            )

            # SES should converge to AES after sufficient samples
            ses_1 = expected_shortfall(volumes, dense_dist)
            # this requires a large number of samples due to poor convergence in the
            # covariance matrix
            aes_6 = expected_shortfall(volumes, rand(dense_dist, 1_000_000))
            @test isapprox(ses_1, aes_6, atol=1e-1)

        end

        @testset "erroring" begin

            returns = collect(1:100)
            risk_level = 1  # wants Float64
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)
            risk_level = 0  # wants Float64
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)
            risk_level = 1.0
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)
            risk_level = 0.0
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)
            risk_level = -0.5
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)
            risk_level = 1.1
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)

            returns = []
            risk_level = 1/2
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)
            returns = [1]
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)
            returns = collect(1:100)
            risk_level = 1/101
            @test_throws ArgumentError expected_shortfall(returns; risk_level=risk_level)

            # wrong number of args elements in financial function calls
            volumes = [-1, 2, -3]
            delta_dist = MvNormal(ones(3))
            supply_pi = [0.1, 0.1, 0.1]
            demand_pi = [0.1, 0.1, 0.1]

            deltas = mean(delta_dist)
            samples = rand(delta_dist, 3)

            bad_args = (volumes, supply_pi, demand_pi)  # length(bad_args) exceeds limit of 3

            # expected return
            @test_throws AssertionError expected_return(volumes, deltas, bad_args...)
            @test_throws AssertionError expected_return(volumes, samples, bad_args...)
            @test_throws AssertionError expected_return(volumes, delta_dist, bad_args...)

            # sharpe ratio
            @test_throws AssertionError sharpe_ratio(volumes, samples, bad_args...)

            # expected shortfall
            @test_throws AssertionError expected_shortfall(volumes, deltas, bad_args...)
            @test_throws AssertionError expected_shortfall(volumes, samples, bad_args...)
            @test_throws AssertionError expected_shortfall(volumes, delta_dist, bad_args...)

        end
    end
end
