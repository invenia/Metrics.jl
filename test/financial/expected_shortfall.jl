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

        @testset "per MW ES" begin
            returns = randn(100)
            volumes = rand(100)

            # Check that it is extensive
            @test expected_shortfall(returns, per_MW=true, volumes=volumes) ≈
                expected_shortfall(2 .* returns, per_MW=true, volumes=2 .* volumes)
            @test 2 * expected_shortfall(returns, per_MW=true, volumes=volumes) ≈
                expected_shortfall(2 .* returns, per_MW=true, volumes=volumes)

            # Check error
            @test_throws ArgumentError expected_shortfall(returns, per_MW=true)
            @test_throws ArgumentError expected_shortfall(returns, per_MW=true, volumes=rand(5))
        end
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
        @test expected_shortfall(volumes, samples; risk_level=0.01) === missing

        # single sample should not work given `risk_level=1` but not otherwise.
        @test_throws MethodError expected_shortfall([-5], [10]; risk_level=0.99)

    end

    @testset "analytic ES" begin
        seed!(1)
        volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
        dense_dist = generate_mvnormal(10)
        nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))

        names = "node" .* string.(collect(1:10))
        dense_id = IndexedDistribution(dense_dist, names)

        @testset "with $type" for (type, dist) in (
            ("Distribution", dense_dist),
            ("IndexedDistribution", dense_id)
        )
            # basic usage
            expected = 26.995589121396023
            @test expected_shortfall(volumes, dist; risk_level=0.5) ≈ expected

            expected = 73.06492436615295
            @test expected_shortfall(volumes, dist; risk_level=0.01) ≈ expected
            @test evaluate(expected_shortfall, volumes, dist; risk_level=0.01) ≈ expected

            # with price impact ES should increase (due to sign)
            @test isless(
                expected,
                expected_shortfall(volumes, dist, nonzero_pi...; risk_level=0.01),
            )
            @test isless(
                expected,
                evaluate(expected_shortfall, volumes, dist, nonzero_pi...; risk_level=0.01),
            )

            # SES should converge to AES after sufficient samples
            ses_1 = expected_shortfall(volumes, dist)
            # this requires a large number of samples due to poor convergence in the
            # covariance matrix
            aes_6 = expected_shortfall(volumes, rand(dist, 1_000_000))
            @test isapprox(ses_1, aes_6, atol=1e-1)
        end
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

        @testset "insufficient samples" begin
            returns = []
            risk_level = 1/2
            @test expected_shortfall(returns; risk_level=risk_level) === missing
            returns = [1]
            @test expected_shortfall(returns; risk_level=risk_level) === missing
            returns = collect(1:100)
            risk_level = 1/101
            @test expected_shortfall(returns; risk_level=risk_level) === missing
        end

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

        @testset "NaN" begin
            @test isnan(evano(zeros(100); risk_level=risk_level))
            @test isnan(evaluate(evano, zeros(100); risk_level=risk_level))
        end
        @testset "Inf" begin
            returns = collect(1.0:100.0)
            returns[1:25] .= 0.0
            @test isinf(evano(returns; risk_level=risk_level))
            @test isinf(evaluate(evano, returns; risk_level=risk_level))
        end

        @testset "DivisionError" begin
            returns = FixedDecimal{Int, 2}.(zeros(100))
            @test_throws DivideError evano(returns; risk_level=risk_level)
            @test_throws DivideError evaluate(evano, returns; risk_level=risk_level)
        end
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
        @test evano(volumes, samples; risk_level=0.01) === missing

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
