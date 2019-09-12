@testset "divergence.jl" begin
    @testset "kullback_leibler" begin
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

        # Generate a covariance matrix where det(Σ) == 0
        Σ0̂ = PSDMat(cholesky(zeros(3, 3), Val(true); check=false))

        # Generate a covariance matrix where Σ = 0̂
        m0̂  = Matrix(zeros(3, 3))

        @testset "equal distributions" begin
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

        @testset "is not symmetric" begin
            # The kullback leibler divergence of p||q is not equal to the kullback leibler
            # divergence of q||p
            p = MvNormal(μ0, Σ0)
            q = MvNormal(μ0, Σ1)

            @test kullback_leibler(p, q) != kullback_leibler(q, p)
        end

        @testset "equal means - varying covariances" begin
            # Diagonal
            p = MvNormal(μ0, Σ0)
            q = MvNormal(μ0, Σ1)

            expected = 0.5147665833421128
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected

            # Random
            p = generate_mvnormal(μ0, 3)
            q = generate_mvnormal(μ0, 3)

            expected = 42.34953520040855
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected
        end

        @testset "equal covariances - varying means" begin
            # Diagonal
            p = MvNormal(μ0, Σ0)
            q = MvNormal(μ1, Σ0)

            expected = 460.625
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected

            # Random
            p = MvNormal(rand(3), Σ0)
            q = MvNormal(rand(3), Σ0)

            expected = 0.02911923964651164
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected
        end

        @testset "different distribution dimensions" begin
            # Both distributions p and q must have the same distribution dimension count, if
            # they don't an error will be raised
            d0 = rand(3)
            d1 = rand(4)
            p = MvNormal(rand(3), d0' * d0)
            q = MvNormal(rand(4), d1' * d1)

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
            end

            @testset "μ0 != μ1 and Σ0 = Σ1 = 1̂ (and is symmetric)" begin
                p = MvNormal(μ0, Σ1)
                q = MvNormal(μ1, Σ1)
                kl = kullback_leibler(p, q)

                @test_broken 0.5 * dot(μdiff, μdiff) ≈ kl
                @test kl < kl_full
            end

            @testset "μ0 = μ1 and Σ0 = Σ1 = 1̂ (and is symmetric)" begin
                p = MvNormal(μ0, Σ0)
                kl = kullback_leibler(p, p)

                @test isapprox(0, kl; atol=1e-12)
                @test kl < kl_full
            end

            @testset "μ0 != μ1 and Σ0 != 0̂ and Σ1 = 0̂ (this test should error)" begin
                p = MvNormal(μ0, Σ0)
                q = MvNormal(μ1, Σ0̂)

                @test_throws ArgumentError kullback_leibler(p, q)
                @test 0.5 * (tr(m0̂) - k + log(det(m0̂) / det(Σ0))) == -Inf
            end

            @testset "μ0 != μ1 and Σ0 = 0̂ and Σ1 != 0̂ (this test should error)" begin
                p = MvNormal(μ0, Σ0̂)
                q = MvNormal(μ1, Σ1)

                @test_throws ArgumentError kullback_leibler(p, q)
            end

            @testset "μ0 != μ1 and Σ0 = Σ1 = 0̂ (this test should especially error)" begin
                p = MvNormal(μ0, Σ0̂)
                q = MvNormal(μ1, Σ0̂)

                @test_throws ArgumentError kullback_leibler(p, q)
            end
        end

        @testset "simple examples" begin
            # Exploiting the simple case of diagonal covariances and zero means
            diag0 = rand(0.1:1e-10:1.0, 10).^2
            diag1 = rand(0.1:1e-10:1.0, 10).^2
            p = MvNormal(zeros(10), Diagonal(diag0))
            q = MvNormal(zeros(10), Diagonal(diag1))

            @test kullback_leibler(p, q) ≈ 0.5 * (
                sum(diag0 ./ diag1) - 10 + log(prod(diag1)/prod(diag0))
            )

            diag0 = rand(0.1:1e-10:1.0, 10).^2
            m0 = rand(10)
            m1 = rand(10)
            p = MvNormal(m0, Diagonal(diag0))
            q = MvNormal(m1, I)

            @test kullback_leibler(p, q) ≈ 0.5 * (
                sum(diag0) + dot(m1 .- m0, m1 .- m0) - 10 - sum(log.(diag0))
            )
        end

        @testset "Univariate Distributions" begin
            p = Normal(0., 1.)
            q = Normal(1., 2.)

            expected = 0.4431471805599453
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected
        end
    end
end
