@testset "divergence.jl" begin
    @testset "kullback_leibler" begin
        @testset "equal distributions" begin
            # If both distributions p are the same, the divergence should be 0

            # Using diagonal cov matrix
            diag_sqrtcov = Diagonal([5.0, 6.0, 7.0])
            p = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)

            expected = 0.0
            @test kullback_leibler(p, p) ≈ expected
            @test evaluate(kullback_leibler, p, p) ≈ expected
        end

        @testset "different distributions" begin
            # If two distributions p and q are different, the kullback_leibler divergence
            # will be greater than 0
            # Using diagonal cov matrix
            # Small divergence example
            diag_sqrtcov = Diagonal([5.0, 6.0, 7.0])
            p = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)
            diag_sqrtcov = Diagonal([6.0, 6.0, 7.0])
            q = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)

            expected = 0.02954377901617672
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected

            # Large divergence example
            diag_sqrtcov = Diagonal([5.0, 6.0, 7.0])
            p = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)
            diag_sqrtcov = Diagonal([1.0, 2.0, 3.0])
            q = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)

            expected = 14.666874160732808
            @test kullback_leibler(p, q) ≈ expected
            @test evaluate(kullback_leibler, p, q) ≈ expected
        end

        @testset "difference dimensions" begin
            # Both distributions p and q must have the same distribution dimension count, if
            # they don't an error will be raised

            diag_sqrtcov = Diagonal([5.0, 6.0, 7.0])
            p = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)
            diag_sqrtcov = Diagonal([5.0, 6.0, 7.0, 8.0])
            q = MvNormal([0.0, 1.0, 10.0, 100.0], diag_sqrtcov' * diag_sqrtcov)

            @test_throws DimensionMismatch kullback_leibler(p, q)
        end
    end
end
