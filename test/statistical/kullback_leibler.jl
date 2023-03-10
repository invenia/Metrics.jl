@testset "kullback_leibler.jl" begin
    # Test Means
    μ0 = [0., 1., 10.]
    μ1 = [1., 10., 100.]
    μdiff = μ0 .- μ1

    # Test Diagonal Matricies
    d0 = Diagonal([1., 2., 3.])
    d1 = Diagonal([2., 3., 4.])

    # Test Covariance Matrices
    Σ0 = d0' * d0
    Σ1 = d1' * d1

    # Distribution Dimention
    k = 3

    # Proper full KL calculation to test certain cases against
    kl_full = kullback_leibler(MvNormal(μ0, Σ0), MvNormal(μ1, Σ1))

    # KL Full should be greater than 0
    @test kl_full > 0

    # Generate a covariance matrix where det(Σ) == 0
    Σ_singular = PSDMat(cholesky(zeros(3, 3), Val(true); check=false))

    # Generate a covariance matrix where Σ = 0̂
    m0̂  = Matrix(zeros(3, 3))

    @testset "is not symmetric" begin
        # The kullback leibler divergence of p||q is not equal to the kullback leibler
        # divergence of q||p
        p = MvNormal(μ0, Σ0)
        q = MvNormal(μ0, Σ1)

        @test kullback_leibler(p, q) != kullback_leibler(q, p)
    end

    @testset "different distribution dimensions" begin
        rng = MersenneTwister(1)
        # Both distributions p and q must have the same distribution dimension count, if
        # they don't an error will be raised
        d0 = rand(rng, 3)
        d1 = rand(rng, 4)
        p = MvNormal(rand(rng, 3), d0' * d0)
        q = MvNormal(rand(rng, 4), d1' * d1)

        @test_throws DimensionMismatch kullback_leibler(p, q)
    end

    @testset "edge cases" begin
        @testset "μ0 = μ1 and Σ0 != Σ1" begin
            p = MvNormal(μ0, Σ0)
            q = MvNormal(μ0, Σ1)
            kl = kullback_leibler(p, q)

            @test 0.5 * (tr(Σ1 \ Σ0) - k + log(det(Σ1) / det(Σ0))) ≈ kl
            @test kl < kl_full
        end

        @testset "μ0 != μ1 and Σ0 = Σ1 (and is symmetric)" begin
            p = MvNormal(μ0, Σ1)
            q = MvNormal(μ1, Σ1)
            kl = kullback_leibler(p, q)

            @test 0.5 * μdiff' * (Σ1 \ μdiff) ≈ kl
            @test kl < kl_full
            @test kl ≈ kullback_leibler(q, p)
        end

        @testset "μ0 = μ1 and Σ0 = Σ1 (and is symmetric)" begin
            # If both distributions are the same, the divergence should be 0
            p = MvNormal(μ0, Σ0)
            q = generate_mvnormal(10)
            expected = 0

            # Using diagonal cov matrix
            @test isapprox(kullback_leibler(p, p), expected; atol=1e-12)
            @test isapprox(evaluate(kullback_leibler, p, p), expected; atol=1e-12)

            # Using random symmetric MvNormal
            @test isapprox(kullback_leibler(q, q), expected; atol=1e-12)
            @test isapprox(evaluate(kullback_leibler, q, q), expected; atol=1e-12)
        end

        @testset "μ0 != μ1 and Σ0 = Σ1 = 1̂ (and is symmetric)" begin
            p = MvNormal(μ0, I)
            q = MvNormal(μ1, I)
            kl = kullback_leibler(p, q)

            @test 0.5 * dot(μdiff, μdiff) ≈ kl
            @test kl ≈ kullback_leibler(q, p)

            # In this instance kl is greater than kl_full
            @test kl >= kl_full
        end

        @testset "μ0 = μ1 and Σ0 = Σ1 = 1̂ (and is symmetric)" begin
            p = MvNormal(μ0, I)
            kl = kullback_leibler(p, p)

            @test isapprox(0, kl; atol=1e-12)
            @test kl < kl_full
        end

        @testset "μ0 != μ1 and Σ0 != 0̂ and Σ1 = 0̂" begin
            p = MvNormal(μ0, Σ0)
            q = MvNormal(μ1, Σ_singular)

            @test_throws ArgumentError kullback_leibler(p, q)
            @test 0.5 * (tr(m0̂) - k + log(det(m0̂) / det(Σ0))) == -Inf
        end

        @testset "μ0 != μ1 and Σ0 = 0̂ and Σ1 != 0̂" begin
            p = MvNormal(μ0, Σ_singular)
            q = MvNormal(μ1, Σ1)

            @test_throws ArgumentError kullback_leibler(p, q)
        end

        @testset "μ0 != μ1 and Σ0 = Σ1 = 0̂" begin
            p = MvNormal(μ0, Σ_singular)
            q = MvNormal(μ1, Σ_singular)

            @test_throws ArgumentError kullback_leibler(p, q)
        end
    end

    @testset "simple examples" begin
        rng = MersenneTwister(1)
        # Exploiting the simple case of diagonal covariances and zero means
        diag0 = rand(rng, 0.1:1e-3:1.0, 10).^2
        diag1 = rand(rng, 0.1:1e-3:1.0, 10).^2
        p = MvNormal(zeros(10), Diagonal(diag0))
        q = MvNormal(zeros(10), Diagonal(diag1))

        @test kullback_leibler(p, q) ≈ 0.5 * (
            sum(diag0 ./ diag1) - 10 + log(prod(diag1) / prod(diag0))
        )

        diag0 = rand(rng, 0.1:1e-3:1.0, 10).^2
        m0 = rand(rng, 10)
        m1 = rand(rng, 10)
        p = MvNormal(m0, Diagonal(diag0))
        q = MvNormal(m1, I)

        @test kullback_leibler(p, q) ≈ 0.5 * (
            sum(diag0) + dot(m1 .- m0, m1 .- m0) - 10 - sum(log.(diag0))
        )
    end

    @testset "Univariate Distributions" begin
        p = Normal(0., 1.)
        q = Normal(1., 2.)

        expected = 0.5 * ((q.σ^2 \ p.σ^2) + (q.μ - p.μ)^2 / q.σ^2 - 1 + log(q.σ^2 / p.σ^2))
        @test kullback_leibler(p, q) ≈ expected
        @test evaluate(kullback_leibler, p, q) ≈ expected
    end

    @testset "using KeyedDistributions" begin
        names = ["foo", "bar", "baz"]
        rng = MersenneTwister(1)

        @testset "normal usage" begin

            a = KeyedDistribution(MvNormal(μ0, Σ0), names)
            b = KeyedDistribution(MvNormal(μ1, Σ1), names)

            expected = kullback_leibler(distribution(a), distribution(b))
            @test kullback_leibler(a, b) == expected
            @test evaluate(kullback_leibler, a, b) == expected

        end
        @testset "shuffled" begin
            a = KeyedDistribution(MvNormal(μ0, Σ0), shuffle(rng, names))
            b = KeyedDistribution(MvNormal(μ1, Σ1), shuffle(rng, names))

            ida = only(axiskeys(a))
            idb = only(axiskeys(b))

            pa = sortperm(ida)
            pb = sortperm(idb)

            sorted_a = MvNormal(μ0[pa], Matrix(Σ0)[pa, pa])
            sorted_b = MvNormal(μ1[pb], Matrix(Σ1)[pb, pb])

            expected = kullback_leibler(sorted_a, sorted_b)
            @test kullback_leibler(a, b) == expected
            @test evaluate(kullback_leibler, a, b) == expected

        end

        @testset "erroring" begin
            # dimensions don't match
            a = KeyedDistribution(MvNormal(rand(4), 1), ["a", "b", "c", "d"])
            b = KeyedDistribution(MvNormal(μ1, Σ1), ["a", "b", "c"])
            @test_throws DimensionMismatch kullback_leibler(a, b)

            # obs lists are different
            a = KeyedDistribution(MvNormal(μ0, Σ0), ["a", "b", "c"])
            b = KeyedDistribution(MvNormal(μ1, Σ1), ["a", "b", "q"])
            @test_throws ArgumentError kullback_leibler(a, b)

            # only works on gaussians
            a = KeyedDistribution(MvNormal(μ0, Σ0), ["a", "b", "c"])
            b = KeyedDistribution(MvLogNormal(μ1), ["a", "b", "c"])
            @test_throws MethodError kullback_leibler(a, b)
        end
    end
end
