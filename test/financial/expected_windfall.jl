@testset "expected windfall" begin
    @testset "simple EW" begin
        # basic usage
        returns = collect(1:100)
        rng = StableRNG(1)

        # default level is 0.05
        @test expected_windfall(returns) == mean(96:100)

        # order shouldn't matter
        shuffle!(rng, returns)
        @test expected_windfall(returns) == mean(96:100)

        # testing a different level
        level = 0.25
        @test expected_windfall(returns; level=level) == mean(76:100)
        @test evaluate(expected_windfall, returns; level=level) == mean(76:100)

        # replacing smaller values shouldn't matter
        returns_2 = sort(returns)
        returns_2[1:95] .= -1000
        shuffle!(rng, returns_2)
        @test expected_windfall(returns) == expected_windfall(returns_2)

        # hard shifts should shift the result
        @test expected_windfall(returns) + 1000 == expected_windfall(returns .+ 1000)

        # should be linear on scalar multiplication
        @test expected_windfall(2 .* returns) == 2 * expected_windfall(returns)

        @testset "per MW ES" begin
            returns = randn(rng, 100)
            volumes = rand(rng, 100)

            # Check that it is extensive
            @test expected_windfall(returns, per_mwh=true, volumes=volumes) ≈
                expected_windfall(2 .* returns, per_mwh=true, volumes=2 .* volumes)
            @test 2 * expected_windfall(returns, per_mwh=true, volumes=volumes) ≈
                expected_windfall(2 .* returns, per_mwh=true, volumes=volumes)

            # Check error
            @test_throws ArgumentError expected_windfall(returns, per_mwh=true)
            @test_throws DimensionMismatch expected_windfall(returns, per_mwh=true, volumes=rand(rng, 5))
        end
    end

    @testset "sample EW" begin
        # using samples
        rng = StableRNG(1234)
        volumes = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
        samples = Matrix(I, (10, 10))
        expected = mean([1, 3, 5, 7, 9])
        @test expected_windfall(volumes, samples; level=0.5) == expected
        @test expected_windfall(volumes, samples; per_mwh=true, level=0.5) == 0.09090909090909091
        @test evaluate(expected_windfall, volumes, samples; level=0.5) == expected

        # using diagonal matrix of samples - requires AbstractArray
        sample_deltas = Diagonal(1:10)
        expected =  mean([1, 3, 5, 7, 9].^2)
        @test expected_windfall(volumes, sample_deltas; level=0.5) == expected

        # generate samples from distribution of deltas
        volumes = repeat([1, -2, 3, -4, 5, -6, 7, -8, 9, -10], 2)
        samples = rand(rng, MvNormal(ones(20)), 50)
        nonzero_pi = (supply_pi=fill(0.1, 20), demand_pi=fill(0.1, 20))

        expected = 49.568769808266644
        @test expected_windfall(volumes, samples) ≈ expected
        @test expected_windfall(volumes, samples; per_mwh=true) == 0.4506251800751513
        @test evaluate(expected_windfall, volumes, samples; obsdim=2) ≈ expected

        # with price impact EW should decrease (due to sign)
        @test expected_windfall(volumes, samples, nonzero_pi...) < expected
        @test isless(
            evaluate(expected_windfall, volumes, samples, nonzero_pi...; obsdim=2),
            expected,
        )

        # too few samples
        @test expected_windfall(volumes, samples; level=0.01) === missing

        # single sample should not work given `level=1` but not otherwise.
        @test_throws MethodError expected_windfall([-5], [10]; level=0.99)

    end

    @testset "erroring" begin
        rng = StableRNG(1)
        returns = collect(1:100)
        for level in (1, 0, 1.0, 0.0, -0.5, 1.1)
            @test_throws ArgumentError expected_windfall(returns; level=level)
        end

        @testset "insufficient samples" begin
            returns = []
            level = 1/2
            @test expected_windfall(returns; level=level) === missing
            returns = [1]
            @test expected_windfall(returns; level=level) === missing
            returns = collect(1:100)
            level = 1/101
            @test expected_windfall(returns; level=level) === missing
        end

        # wrong number of args elements in financial function calls
        volumes = [-1, 2, -3]
        delta_dist = MvNormal(ones(3))
        supply_pi = [0.1, 0.1, 0.1]
        demand_pi = [0.1, 0.1, 0.1]

        deltas = mean(delta_dist)
        samples = rand(rng, delta_dist, 3)

        bad_args = (volumes, supply_pi, demand_pi)  # length(bad_args) exceeds limit of 3

        # expected windfall
        @test_throws MethodError expected_windfall(volumes, deltas, bad_args...)
        @test_throws MethodError expected_windfall(volumes, samples, bad_args...)
        @test_throws MethodError expected_windfall(volumes, delta_dist, bad_args...)
    end

    @testset "ew_over_es" begin
        rng = StableRNG(1)
        returns = randn(rng, 100)

        @test ew_over_es(returns) ≈ expected_windfall(returns) / expected_shortfall(returns)

        volumes = rand(rng, 100)

        @test ew_over_es(returns, per_mwh=true, volumes=volumes) ≈
            expected_windfall(returns, per_mwh=true, volumes=volumes) /
            expected_shortfall(returns, per_mwh=true, volumes=volumes)

        # Check that it is invariant under scaling
        @test ew_over_es(returns, per_mwh=true, volumes=volumes) ≈
            ew_over_es(2 .* returns, per_mwh=true, volumes=2 .* volumes)
        @test ew_over_es(returns, per_mwh=true, volumes=volumes) ≈
            ew_over_es(2 .* returns, per_mwh=true, volumes=volumes)

        # Check error
        @test_throws ArgumentError ew_over_es(returns, per_mwh=true)
        @test_throws DimensionMismatch ew_over_es(returns, per_mwh=true, volumes=rand(rng, 5))
    end
end
