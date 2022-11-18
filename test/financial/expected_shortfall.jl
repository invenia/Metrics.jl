@testset "expected shortfall" begin
    rng = StableRNG(1234)
    i_matrix_samples = Matrix(I, (10, 10))
    mv_normal_samples = rand(rng, MvNormal(ones(20)), 50)
    @testset "simple ES" begin
        # basic usage
        returns = collect(1:100)

        # default risk_level is 0.05
        @test expected_shortfall(returns) == -3

        # order shouldn't matter
        shuffle!(rng, returns)
        @test expected_shortfall(returns) == -3

        # testing a different risk_level
        risk_level = 0.25
        @test expected_shortfall(returns; risk_level=risk_level) == -13
        @test evaluate(expected_shortfall, returns; risk_level=risk_level) == -13

        # replacing larger values shouldn't matter
        returns_2 = sort(returns)
        returns_2[6:end] .= 1000
        shuffle!(rng, returns_2)
        @test expected_shortfall(returns) == expected_shortfall(returns_2)

        @testset "per MW ES" begin
            returns = randn(rng, 100)
            volumes = rand(rng, 100)

            # Check that it is extensive
            @test expected_shortfall(returns, per_mwh=true, volumes=volumes) ≈
                expected_shortfall(2 .* returns, per_mwh=true, volumes=2 .* volumes)
            @test 2 * expected_shortfall(returns, per_mwh=true, volumes=volumes) ≈
                expected_shortfall(2 .* returns, per_mwh=true, volumes=volumes)

            # Check error
            @test_throws ArgumentError expected_shortfall(returns, per_mwh=true)
            @test_throws DimensionMismatch expected_shortfall(returns, per_mwh=true, volumes=rand(5))
        end
    end

    @testset "sample ES" begin
        # using samples
        volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
        expected = -mean([-2, -4, -6, -8, -10])  # = 6.0
        @test expected_shortfall(volumes, i_matrix_samples; risk_level=0.5) == expected
        @test evaluate(expected_shortfall, volumes, i_matrix_samples; risk_level=0.5) == expected

        # using diagonal matrix of samples - requires AbstractArray
        sample_deltas = Diagonal(1:10)
        expected =  -mean([-2*2, -4*4, -6*6, -8*8, -10*10])  # = 44.0
        @test expected_shortfall(volumes, sample_deltas; risk_level=0.5) == expected

        # generate samples from distribution of deltas
        volumes = repeat([1, -2, 3, -4, 5, -6, 7, -8, 9, -10], 2)
        nonzero_pi = (supply_pi=fill(0.1, 20), demand_pi=fill(0.1, 20))

        expected = 71.644648723418
        @test expected_shortfall(volumes, mv_normal_samples) ≈ expected
        @test evaluate(expected_shortfall, volumes, mv_normal_samples; obsdim=2) ≈ expected

        # with price impact ES should increase (due to sign)
        @test expected_shortfall(volumes, mv_normal_samples, nonzero_pi...) > expected
        @test isless(
            expected,
            evaluate(expected_shortfall, volumes, mv_normal_samples, nonzero_pi...; obsdim=2),
        )

        # too few samples
        @test expected_shortfall(volumes, mv_normal_samples; risk_level=0.01) === missing

        # single sample should not work given `risk_level=1` but not otherwise.
        @test_throws MethodError expected_shortfall([-5], [10]; risk_level=0.99)

    end

    @testset "analytic ES" begin
        # rng = StableRNG(1)
        # for constructing some PDMats
        B = (reshape(2:10, 3, 3) / 12) .^ 2
        A = B' * B + I

        D = Diagonal([1.,2,3])
        S = Diagonal([4.,5,6])

        volumes = [1, 2, -3]
        nonzero_pi = (supply_pi=fill(0.1, length(volumes)), demand_pi=fill(0.1, length(volumes)))

        names = "node" .* string.(collect(1:length(volumes)))

        @testset "AbstractPDMat type $(typeof(pd))" for pd in [
            PDiagMat(diag(D)),
            PDMat(Symmetric(A)),
            # NOTE: # `PSDMat` experienced some sampling issue for now and the empirical results
            #   don't match analytical calculation for now. Exclude `PSDMat` in the test
            #   until this issue (https://github.com/invenia/PDMatsExtras.jl/issues/11) is resolved
            # PSDMat(Symmetric(A)),
            WoodburyPDMat(B, D, S)
        ]
            @testset "distribution type $(typeof(dist))" for dist in [
                MvNormal(ones(size(pd, 1)), pd),
                GenericMvTDist(2.2, ones(size(pd, 1)), pd),
            ]
                # with price impact ES should increase (due to sign)
                @test isless(
                    expected_shortfall(volumes, dist; risk_level=0.01),
                    expected_shortfall(volumes, dist, nonzero_pi...; risk_level=0.01),
                )
                @test isless(
                    evaluate(expected_shortfall, volumes, dist; risk_level=0.01),
                    evaluate(expected_shortfall, volumes, dist, nonzero_pi...; risk_level=0.01),
                )

                # SES should converge to AES after sufficient samples
                aes = expected_shortfall(volumes, dist)
                # this requires a large number of samples due to poor convergence in the
                # covariance Matrixvariate
                rng = StableRNG(1)
                ses = expected_shortfall(volumes, rand(rng, dist, 1_000_000))
                @test isapprox(aes, ses, atol=1e-1)
            end

            # special case when dof is large for T distribution
            @testset "risk level: $α" for α in collect(0.2: 0.3: 0.8)
                mvn = MvNormal(ones(size(pd, 1)), pd)
                mvt = GenericMvTDist(1_000_000, ones(size(pd, 1)), pd)
                @test expected_shortfall(volumes, mvn; risk_level=α) ≈
                    expected_shortfall(volumes, mvt; risk_level=α) atol=1e-3
            end
        end

        @testset "fixed basic performance" begin
            pd = PDMat(Symmetric(A))
            # the `expected` values are for catching future regression
            @testset "MvNormal" begin
                dist = MvNormal(ones(size(pd, 1)), pd)
                idist = KeyedDistribution(dist, names)
                expected = 4.896918621367946
                @test expected_shortfall(volumes, dist; risk_level=0.3) ≈ expected
                @test expected_shortfall(volumes, idist; risk_level=0.3) ≈ expected
            end
            @testset "MvT" begin
                dist = GenericMvTDist(3.0, ones(size(pd, 1)), pd)
                idist = KeyedDistribution(dist, names)
                expected = 6.971343619802216
                @test expected_shortfall(volumes, dist; risk_level=0.3) ≈ expected
                @test expected_shortfall(volumes, idist; risk_level=0.3) ≈ expected
            end
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
        rng = StableRNG(1)

        deltas = mean(delta_dist)
        samples = rand(rng, delta_dist, 3)

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
    rng = StableRNG(1234)
    i_matrix_samples = Matrix(I, (10, 10))
    mv_normal_samples = rand(rng, MvNormal(ones(20)), 50)
    @testset "simple evano" begin
        # Basic usage
        returns = collect(1:100)

        # Default risk_level is 0.05
        expected = -16.833333333333332
        @test evano(returns) == expected

        # Order shouldn't matter
        rng = StableRNG(1)
        shuffle!(rng, returns)
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
    end

    @testset "sample evano" begin
        # Using samples
        volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
        expected = -0.08333333333333333
        @test evano(volumes, i_matrix_samples; risk_level=0.5) == expected
        @test evaluate(evano, volumes, i_matrix_samples; risk_level=0.5) == expected

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
        volumes = repeat([1, -2, 3, -4, 5, -6, 7, -8, 9, -10], 2)
        expected = -0.040273663897995186
        @test evano(volumes, mv_normal_samples) ≈ expected
        @test evaluate(evano, volumes, mv_normal_samples; obsdim=2) ≈ expected

        # too few samples
        @test evano(volumes, mv_normal_samples; risk_level=0.01) === missing

        # single sample should not work given `risk_level=1` but not otherwise.
        @test_throws MethodError evano([-5], [10]; risk_level=0.99)
    end

    @testset "analytic evano" begin
        # We currently don't have a working version of this for Multivariate
        # Distributions as there are many definitions of `median` which aren't
        # implemented by `Distributions`.
        # https://invenia.slack.com/archives/CMMAKP97H/p1567612804011200?thread_ts=1567543537.008300&cid=CMMAKP97H
        # More info: https://www.r-bloggers.com/multivariate-medians/
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


@testset "mean over es" begin
    rng = StableRNG(1234)
    i_matrix_samples = Matrix(I, (10, 10))
    mv_normal_samples = rand(rng, MvNormal(ones(20)), 50)
    @testset "simple case" begin
        # Basic usage
        rng = StableRNG(1)
        returns = collect(100:200)

        # Default risk_level is 0.05
        expected = -1.4705882352941178
        @test mean_over_es(returns) == expected

        @testset "order shouldn't matter" begin
            shuffle!(rng, returns)
            @test mean_over_es(returns) == expected
        end

        risk_level = 0.25
        @testset "test different risk level" begin
            expected = -1.3392857142857142
            @test mean_over_es(returns; risk_level=risk_level) == expected
            @test evaluate(mean_over_es, returns; risk_level=risk_level) == expected
        end

        @testset "NaN" begin
            @test isnan(mean_over_es(zeros(100); risk_level=risk_level))
            @test isnan(evaluate(mean_over_es, zeros(100); risk_level=risk_level))
        end
        @testset "Inf" begin
            returns = collect(1.0:100.0)
            returns[1:25] .= 0.0
            @test isinf(mean_over_es(returns; risk_level=risk_level))
            @test isinf(evaluate(mean_over_es, returns; risk_level=risk_level))
        end
    end

    volumes = [6, -7, 8, -9, 10, -11, 12, -13, 14, -15]
    @testset "sample mean over es" begin
        @testset "using samples" begin
            samples = Matrix(I, (10, 10))
            expected = -0.045454545454545435
            @test mean_over_es(volumes, samples; risk_level=0.5) == expected
            @test evaluate(mean_over_es, volumes, samples; risk_level=0.5) == expected
        end

        sample_deltas = Diagonal(1:10)
        expected = -0.10810810810810813
        @testset "method works with AbstractArray" begin
            @test mean_over_es(volumes, sample_deltas; risk_level=0.5) == expected
        end

        # with price impact mean over es should decrease since expected shortfall increases
        @testset "mean over es decreases with pi" begin
            nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))
            @test mean_over_es(volumes, sample_deltas, nonzero_pi...; risk_level=0.5) < expected
            @test evaluate(mean_over_es, volumes, sample_deltas, nonzero_pi...; risk_level=0.5, obsdim=2) < expected
        end

        # generate samples from distribution of deltas
        @testset "using samples from delta distribution" begin
            volumes = repeat([1, -2, 3, -4, 5, -6, 7, -8, 9, -10], 2)

            expected = -0.03761927546629896
            @test mean_over_es(volumes, mv_normal_samples) ≈ expected
            @test evaluate(mean_over_es, volumes, mv_normal_samples; obsdim=2) ≈ expected
        end

        @testset "too few samples" begin
            @test mean_over_es(volumes, mv_normal_samples; risk_level=0.01) === missing
        end

        # single sample should not work given `risk_level=1` but not otherwise.
        @testset "MethodError for single sample" begin
            @test_throws MethodError mean_over_es([-5], [10]; risk_level=0.99)
        end
    end
end

@testset "mean_minus_es" begin
    @testset "simple case" begin
        returns = 1:20

        # Default α is 0.25
        expected = 10.5 - 0.25 * (-1)
        @test mean_minus_es(returns) ≈ expected
        @test evaluate(mean_minus_es, returns) ≈ expected

        @testset "Different risk level" begin
            expected = 10.5 - 0.25 * (-1.5)
            @test mean_minus_es(returns; risk_level=0.1) ≈ expected
            @test evaluate(mean_minus_es, returns; risk_level=0.1) ≈ expected
        end

        @testset "Different alpha" begin
            expected = 10.5 - 0.1 * (-1.5)
            @test mean_minus_es(returns; α=0.1, risk_level=0.1) ≈ expected
            @test evaluate(mean_minus_es, returns; α=0.1, risk_level=0.1) ≈ expected
        end

        @testset "Random data" begin
            returns = randn(1000)
            @test mean_minus_es(returns) ≈ mean(returns) - 0.25 * es(returns)
        end
    end

    @testset "volumes and deltas" begin
        volumes = randn(10)
        deltas = randn(10, 100)

        returns = vec(sum(volumes .* deltas; dims=1))

        expected = mean(returns) - 0.25 * es(returns)
        @test mean_minus_es(volumes, deltas) ≈ expected
        @test evaluate(mean_minus_es, volumes, deltas; obsdim=2) ≈ expected

        @testset "Different risk level" begin
            expected = mean(returns) - 0.25 * es(returns; risk_level=0.1)
            @test mean_minus_es(volumes, deltas; risk_level=0.1) ≈ expected
            @test evaluate(mean_minus_es, volumes, deltas; risk_level=0.1, obsdim=2) ≈ expected
        end

        @testset "Different alpha" begin
            expected = mean(returns) - 0.1 * es(returns; risk_level=0.1)
            @test mean_minus_es(volumes, deltas; α=0.1, risk_level=0.1) ≈ expected
            @test evaluate(mean_minus_es, volumes, deltas; α=0.1, risk_level=0.1, obsdim=2) ≈ expected
        end
    end
end
