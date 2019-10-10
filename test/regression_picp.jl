const testing_distributions = let
    sqrtcov4 = randn(4, 4)
    sqrtcov10 = randn(10, 10)
    sqrtcov20 = randn(20, 20)

    (
        # Univariate:
        # NB: We only test on Normal distributions since the tests are easier for these
        # but the code is the same for all univariates so these test cover us sufficiently
        Normal(1,0.2),
        Normal(0.1,8),
        # Multivariate: we **only** support MvNormal)
        MvNormal(3, 0.2),
        MvNormal(randn(4), sqrtcov4 * sqrtcov4'),
        MvNormal(randn(10), sqrtcov10 * sqrtcov10'),
        MvNormal(randn(20), sqrtcov20 * sqrtcov20'),
        # IndexedDistributions: we also **only** support MvNormal
        IndexedDistribution(MvNormal(3, 0.2), ["a", "b", "c"]),
        IndexedDistribution(MvNormal(randn(4), sqrtcov4 * sqrtcov4'), ["a", "b", "c", "d"]),
    )
end

"""
    rand_out_of_dist(dist)

Returns data that is outside the given distribution, for testing purposes.
Generated data comes from a overlapping uniform distribution
Returns an iterator of observations
"""
function rand_out_of_dist(dist, n_samples)
    basis = max((rand(dist) for _ in 1:100)...)
    center = mean(dist)
    return [center .+ 10*(rand()-0.5).*basis for ii in 1:n_samples]
end

@testset "picp" begin
    # Because the confidance region check is a nontrival algorithm (esp for MvNormal)
    # but with very well known known statistical properties
    # We check it here by testing those properties. (rather than checking exact values)
    @testset "Samples based (Univariate only)" begin
        # Testing uniform distribution case
        dist_samples = rand(10_000)
        test_samples = rand(10_000)
        for α in (0.1, 0.3, 0.5, 0.7, 0.9)
            @test picp(α, dist_samples, dist_samples) ≈ α
            @test picp(α, dist_samples, test_samples) ≈ α rtol=0.15

            old_picp = α
            # We were from the same region (0-1 uniform, as we shift away that overlap
            # with the center should start to decrease at 0.5)
            for shift in (0.2, 0.3, 0.5, 1.2, 2, 5)
                shifted_samples = test_samples .+ shift

                new_picp = picp(α, dist_samples, shifted_samples)
                if shift < 0.5 # overlap the same
                    @test new_picp ≈ old_picp rtol=0.25
                elseif shift >= 1  # overlap gone
                    @test new_picp == 0
                else # between 0.5 and 1.0 so overlap decreasing
                    @test new_picp <= old_picp * 1.25 # scaling to allow rtol=0.25
                end
                old_picp = new_picp
            end
        end
    end

    @testset "Distributions - $(typeof(dist))" for dist in testing_distributions
        samples = [rand(dist) for _ in 1:5_000]
        for α in (0.1, 0.3, 0.5, 0.7, 1.0)
            # It is exected that for a good sample the portion of points within the confidance interval
            # will be equal to the size of the confidance interval (α)
            # (Kinda by definition)
            @test picp(α, dist, samples) ≈ α rtol=0.15

            @testset "shifted samples" begin
                # since this is symetrical, and the sample is as good as it gets
                # shifting the sample should make the PICP go down.
                old_picp = α
                for shift in (0.0, 0.2, 0.3, 0.5, 1.2, 2, 5)
                    if dist isa UnivariateDistribution
                        shifted_samples = samples .+ shift
                    else # Multivariate
                        offset = fill(shift, length(dist))
                        shifted_samples = samples .+ Ref(offset)
                    end

                    new_picp = picp(α, dist, shifted_samples)

                    @test new_picp <= old_picp * 1.15 # scaling to allow rtol=0.15
                    old_picp = new_picp
                end
            end  # shifted samples

            @testset "different form of same distribution implies  same PICP" begin
                data = rand_out_of_dist(dist, 5_000)
                @test picp(α, canonform(dist), data) ≈ picp(α, dist, data) rtol=0.05
            end

            @testset "increasing how how different true data is decreases PICP" begin
                old_picp = α
                plain_samples = [rand(dist) for _ in 1:6000]
                corrupt_samples = rand_out_of_dist(dist, 6_000)
                for n_plain in 6000:-1000:0
                    data = vcat(
                        plain_samples[1:n_plain],
                        corrupt_samples[1+n_plain:end]
                    )
                    new_picp = picp(α, dist, data)
                    @test new_picp <= old_picp * 1.25 # scaling to allow rtol=0.25
                end
            end
        end  # different α
    end  # @testset Distributions
end # @testset picp

@testset "wpicp" begin
    # Most of the true functionality is tested in picp tests these tests just make sure we stay
    # in agreement. Particularly for any the optimized cases.

    @testset "Samples based (Univariate only)" begin
        # Testing uniform distribution case
        dist_samples = rand(1000)
        test_samples = rand(1000)
        @test isequal(
            [picp(x, dist_samples, test_samples) for x in 0.2:0.1:0.7],
            wpicp(0.2:0.1:0.7, dist_samples, test_samples)
        )
    end

    @testset "Distributions - $typeof(dist)" for dist in testing_distributions
        ideal_samples =
        @testset "$name" for (name, samples) in (
            ("ideal_samples",  [rand(dist) for _ in 1:5_000]),
            ("nonideal_samples",  rand_out_of_dist(dist, 5_000)),
        )
            @test isequal(
                [picp(x, dist, samples) for x in 0.2:0.1:0.7],
                wpicp(0.2:0.1:0.7, dist, samples)
            )

            # Test default
            @test wpicp(0.1:0.05:0.95, dist, samples) == wpicp(dist, samples)
        end
    end  # @testset Distributions
end # @testset wpicp


@testset "apicp" begin
    # These tests need to show the hopefully useful statistical properties of apicp.
    # since it our own invention

    @testset "Normal Distribution" begin
        @testset "ideal estimate" begin
            # The following invarients are promised only for ideal estimates
            @testset "invarient to mean and spread" begin
                true_points = rand(Normal(0, 0.5), 1_000_000)
                estimated_dist = Normal(0, 0.5)
                @test apicp(estimated_dist, true_points) ≈ 1 atol=0.01

                true_points = rand(Normal(3, 0.5), 1_000_000)
                estimated_dist = Normal(3, 0.5)
                @test apicp(estimated_dist, true_points) ≈ 1 atol=0.01


                true_points = rand(Normal(3, 20), 1_000_000)
                estimated_dist = Normal(3, 20)
                @test apicp(estimated_dist, true_points) ≈ 1 atol=0.01
            end
            @testset "invarient (reasonable) window selection" begin
                true_points = rand(Normal(3, 20), 1_000_000)
                estimated_dist = Normal(3, 20)
                @test apicp(0.05:0.05:0.95, estimated_dist, true_points) ≈ 1 atol=0.01

                true_points = rand(Normal(3, 20), 1_000_000)
                estimated_dist = Normal(3, 20)
                @test apicp(0.05:0.005:0.95, estimated_dist, true_points) ≈ 1 atol=0.01
            end
        end
    end

    @testset "shifted estimate" begin
        true_points = rand(Normal(0, 0.5), 1_000_000)
        @testset "incorrect mean causes decrease" begin
            @test apicp(Normal(1, 0.5), true_points) < apicp(Normal(0, 0.5), true_points)
        end

        @testset "overly narrow distribution causes decrease" begin
            @test apicp(Normal(0, 0.1), true_points) < apicp(Normal(0, 0.5), true_points)
        end

        @testset "overly wide distribution causes increase" begin
            @test apicp(Normal(0, 2), true_points) > apicp(Normal(0, 0.5), true_points)
        end

        @testset "incorrect mean causes symetric decrease (for symetric dist)" begin
            @test ≈(
                apicp(Normal(1, 0.5), true_points),
                apicp(Normal(-1, 0.5), true_points); rtol=0.01
            )
        end
    end
end # @testset apicp
