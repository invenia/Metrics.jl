@testset "bky_test.jl" begin

    pvalues = collect(0.01:0.01:0.1)

    @testset "q ∉ [0, 1]" begin
        @test_throws DomainError bky_test(pvalues, -1)
        @test_throws DomainError bky_test(pvalues, 0)
        @test_throws DomainError bky_test(pvalues, 1)
    end

    @testset "Extreme q within confidence bounds" begin
        @test all(.!bky_test(pvalues, 0.001))  # q is very small - don't reject any H0
        @test all(bky_test(pvalues, 0.2))  # q is very large - reject all H0
    end

    # Example from "Adaptive linear step-up procedures that control the false
    # discovery rate", Benjamini Y., Krieger A.M., Yekutieli D., 2006.
    @testset "paper example" begin
        rng = MersenneTwister(1)

        pvalues = [
            0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
            0.0278, 0.0298, 0.0344, 0.0459, 0.3240,
            0.4262, 0.5719, 0.6528, 0.7590, 1.0,
        ]

        shuffled_pvalues = shuffle(rng, pvalues)

        # returns true/false if null hypothesis is rejected
        rejected = bky_test(shuffled_pvalues)
        @test evaluate(bky_test, shuffled_pvalues) == rejected

        @test sort(shuffled_pvalues[rejected]) == pvalues[1:9]
    end
end
