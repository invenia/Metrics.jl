@testset "divergence.jl" begin
    @testset "kullback_leibler" begin
        @testset "equal distributions" begin
            # If both distributions p are the same, the divergence should be 0

            # Using diagonal cov matrix
            diag_sqrtcov = Diagonal([5., 6., 7.])
            p = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)

            expected = 0.0
            @test kullback_leibler(p, p) ≈ expected
            @test evaluate(kullback_leibler, p, p) ≈ expected

            # Using random symmetric MvNormal
            q = generate_mvnormal(10)
            @test_broken kullback_leibler(q, q) ≈ expected
            @test_broken evaluate(kullback_leibler, q, q) ≈ expected
        end

        @testset "KL(p||q) != KL(q||p)" begin
            # The kullback leibler divergence of p||q is not equal to the kullback leibler
            # divergence of q||p
            diag_sqrtcov = Diagonal([5., 6., 7.])
            p = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)
            diag_sqrtcov = Diagonal([6., 6., 7.])
            q = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)

            @test kullback_leibler(p, q) != kullback_leibler(q, p)
        end

        @testset "different distributions" begin
            # If two distributions p and q are different, the kullback_leibler divergence
            # will be greater than 0
            # Using diagonal cov matrix
            # Small divergence example
            diag_sqrtcov = Diagonal([5., 6., 7.])
            p = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)
            diag_sqrtcov = Diagonal([6., 6., 7.])
            q = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)

            expected = 0.02954377901617672
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected

            # Large divergence example
            diag_sqrtcov = Diagonal([5., 6., 7.])
            p = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)
            diag_sqrtcov = Diagonal([1., 2., 3.])
            q = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)

            expected = 14.666874160732808
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected

            # Using a random symmetric MvNormal
            p = generate_mvnormal(10)
            q = generate_mvnormal(10)

            expected = 80.52297745451973
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected
        end

        @testset "difference dimensions" begin
            # Both distributions p and q must have the same distribution dimension count, if
            # they don't an error will be raised

            diag_sqrtcov = Diagonal([5., 6., 7.])
            p = MvNormal([0., 1., 10.], diag_sqrtcov' * diag_sqrtcov)
            diag_sqrtcov = Diagonal([5., 6., 7., 8.])
            q = MvNormal([0., 1., 10., 100.], diag_sqrtcov' * diag_sqrtcov)

            @test_throws DimensionMismatch kullback_leibler(p, q)
        end

        @testset "edge cases" begin
            @testset "μ0 != μ1 where Σ0 = Σ1" begin
                cov = Diagonal([1., 2., 3.])
                p = MvNormal([0., 1., 10.], cov' * cov)
                q = MvNormal([1., 10., 0.], cov' * cov)

                expected = 16.180555555555554
                @test kullback_leibler(p, q) ≈ expected
                @test evaluate(kullback_leibler, p, q) ≈ expected
            end

            @testset "μ0 != μ1 where Σ0 != Σ1" begin
                cov = Diagonal([1., 2., 3.])
                p = MvNormal([0., 1., 10.], cov' * cov)
                cov = Diagonal([3., 2., 1.])
                q = MvNormal([1., 10., 0.], cov' * cov)

                expected = 63.736111111111114
                @test kullback_leibler(p, q) ≈ expected
                @test evaluate(kullback_leibler, p, q) ≈ expected
            end

            @testset "μ0 = 0 and μ1 != 0 (and the converse)" begin
                cov = Diagonal([1., 2., 3.])
                p = MvNormal([0., 0., 0.], cov' * cov)
                q = MvNormal([1., 10., 0.], cov' * cov)

                expected = 13.0
                @test kullback_leibler(p, q) ≈ expected
                @test evaluate(kullback_leibler, p, q) ≈ expected

                expected = 13.0
                @test kullback_leibler(q, p) ≈ expected
                @test evaluate(kullback_leibler, q, p) ≈ expected
            end

            @testset "Σ0 ≈ 0 and Σ1 != 0 (and the converse)" begin
                # Currently unsure on how to test this
            end

            @testset "Σ0 ≈ Σ1 ≈ 0" begin
                # Currently unsure on how to test this
            end
        end
    end
end
