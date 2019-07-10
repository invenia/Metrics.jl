@testset "picp" begin
    # Because the confidance region check is a nontrival algorithm (esp for MvNormal)
    # but with very well known known statistical properties
    # We check it here by testing those properties. (rather than checking exact values)
    @testset "Samples based (Univariate only)" begin
        # Testing uniform distribution case
        dist_samples = rand(1000)
        test_samples = rand(1000)
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
                    @test new_picp ≈ old_picp rtol=0.2
                elseif shift >= 1  # overlap gone
                    @test new_picp == 0
                else # between 0.5 and 1.0 so overlap decreasing
                    @test new_picp <= old_picp * 1.2 # scaling to allow rtol=0.2
                end
                old_picp = new_picp
            end
        end
    end

    @testset "Distributions" begin
        sqrtcov4 = randn(4, 4)
        sqrtcov10 = randn(10, 10)
        sqrtcov50 = randn(50, 50)
        for dist in (
            # Univariate (We only test on Normal as the test are easier but the code is the
            # same for all univariate distributions, so these test cover us surfiently)
            Normal(1,0.2),
            Normal(0.1,8),
            # Multivariate (We **only** support MvNormal)
            MvNormal(3, 0.2),
            MvNormal(randn(4), sqrtcov4 * sqrtcov4'),
            MvNormal(randn(10), sqrtcov10 * sqrtcov10'),
            MvNormal(randn(50), sqrtcov50 * sqrtcov50'),
        )
            samples = [rand(dist) for _ in 1:5_000]
            for α in (0.1, 0.3, 0.5, 0.7, 1.0)
                # It is exected that for a good sample the portion of points within the confidance interval
                # will be equal to the size of the confidance interval (α)
                # (Kinda by definition)
                @test picp(α, dist, samples) ≈ α rtol=0.15

                # since this is symetrical, and the sample is as good as it gets
                # shifting the sample should make the PICP go down.
                old_picp = α
                for shift in (0.2, 0.3, 0.5, 1.2, 2, 5)
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
            end  # different α
        end  # different distribution parameters
    end  # @testset Distributions
end # @testset picp


#==
@testset "picp" begin
    @testset "base function" begin
        lower_bound = collect(1:10)
        upper_bound = collect(11:20)
        y_pred = collect(6:15)
        @test picp(lower_bound, upper_bound, y_pred) == 1
        lower_bound = collect(1:5)
        upper_bound = collect(11:15)
        y_pred = [2, 14, 1, 4, 15]
        @test picp(lower_bound, upper_bound, y_pred) == 0.6
    end
    @testset "vector point using samples" begin
        α = .5
        y_true = [2.5, -3.74, 5.5]
        samples = [1. 2. 3. 4. 5.; -3. -3.5 -4. -4.5 -5.; -1. 1. 3. 5. 7.]
        @test picp(α, samples, y_true) == 2/3
        α = .25
        @test picp(α, samples, y_true) == 1/3
        α = 1.0
        @test picp(α, samples, y_true) == 3/3
        y_true = [0, -3.76, 5.5]
        @test picp(α, samples, y_true) == 2/3
    end
    @testset "vector point not using samples" begin
        dist = MvNormal([1., 2., 3.], 2.0)
        α = .1
        y_true = [1., -2., -3.]

        seed!(1234)
        @test picp(α, dist, y_true) == 1/3

        dist = MvNormal([1., 2., 3.], sqrt(2.0))
        y_true = [1., 1., -3.]
        α = .6

        seed!(1234)
        @test picp(α, dist, y_true) == 2/3
    end
end

@testset "wpicp" begin
    @testset "base function" begin
        dist = MvNormal([1., 2., 3., 4.], 2.0)
        y_true = [1., -200., -300., -400.]

        seed!(1234)
        @test wpicp(dist, y_true) == fill(.25, 18)

        α_range=0.25:0.05:0.75

        seed!(1234)
        @test wpicp(dist, y_true, α_range) == fill(.25, 11)
        @test length(wpicp(dist, y_true, α_range)) == length(α_range)

        α_min=0.15
        α_max=0.85
        α_step=0.01

        seed!(1234)
        @test wpicp(dist, y_true, α_min=α_min, α_max=α_max, α_step=α_step) ==
            wpicp(dist, y_true, α_min:α_step:α_max)

        y_true = [1., 3., -300., -400.]
        α_range=0.1:0.8:0.9

        seed!(1234)
        @test wpicp(dist, y_true, α_range) == [.25, .5]
    end
end

@testset "apicp" begin
    @testset "base function" begin
        dist = MvNormal([1., 2., 3., 4.], 2.0)
        y_true = [1., 3., -300., -400.]
        α_range=0.1:0.8:0.9

        seed!(1234)
        # dot([0.1, 0.9], [.25, .5]) / sum([0.1, 0.9] .^ 2)
        @test apicp(dist, y_true, α_range) == 0.5792682926829268

        y_true = [1., -200., -300., -400.]
        seed!(1234)
        @test apicp(dist, y_true) == 0.3827460510328068

        α_min=0.15
        α_max=0.85
        α_step=0.01
        seed!(1234)
        @test apicp(dist, y_true, α_min=α_min, α_max=α_max, α_step=α_step) ==
            apicp(dist, y_true, α_min:α_step:α_max)
    end
end
==#
