@testset "regression.jl" begin

    @testset "squared_error" begin
        @testset "scalar point" begin
            y_true = 1
            y_pred = 1
            @test squared_error(y_true, y_pred) == 0
            y_true = 4
            y_pred = 1
            @test squared_error(y_true, y_pred) == 9
            y_true = rand(Int64)
            y_pred = rand(Int64)
            @test squared_error(y_true, y_pred) == squared_error(y_pred, y_true)
        end
        @testset "vector point" begin
            y_true = [1, 2, 3]
            y_pred = [1, 2, 3]
            @test squared_error(y_true, y_pred) == 0
            y_true = [2, 2, 2, 2]
            y_pred = [1, 2, 3, 4]
            @test squared_error(y_true, y_pred) == 6
            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test squared_error(y_true, y_pred) == squared_error(y_pred, y_true)
        end
        @testset "erroring" begin
            y_true = 1
            y_pred = [1, 2, 3]
            @test_throws DimensionMismatch squared_error(y_true, y_pred)
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch squared_error(y_true, y_pred)
        end
    end

    @testset "marginal_loglikelihood" begin
        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_pred = [.1, .2, .3]
            @test marginal_loglikelihood(dist, y_pred) == -4.842507141293854
        end
        @testset "vector point" begin
            dist = MvNormal(3, 2)
            y_pred = [8 10 10; 10 5 7; 9 7 10]
            @test marginal_loglikelihood(dist, y_pred) == -98.00877142388157
        end
    end

    @testset "joint_loglikelihood" begin
        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_pred = [.1, .2, .3]
            @test joint_loglikelihood(dist, y_pred) == -4.842507141293854
            @test joint_loglikelihood(dist, y_pred) == marginal_loglikelihood(dist, y_pred)
        end
        @testset "vector point" begin
            dist = MvNormal(3, 2)
            y_pred = [8 10 10; 10 5 7; 9 7 10]
            @test joint_loglikelihood(dist, y_pred) == -98.00877142388157
        end
    end

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
            # `dist` isn't used
            dist = MvNormal(3, 2)
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
        @testset "vector point not using samples (could fail due to randomness)" begin
            dist = MvNormal([1., 2., 3.], 2.0)
            α = .1
            y_true = [1., -2., -3.]
            @test picp(α, dist, y_true) == 1/3 # This can fail due to randomness
            dist = MvNormal([1., 2., 3.], sqrt(2.0))
            y_true = [1., 1., -3.]
            α = .6
            @test picp(α, dist, y_true) == 2/3 # This can fail due to randomness
        end
    end

    @testset "wpicp" begin
        @testset "base function" begin
            dist = MvNormal([1., 2., 3., 4.], 2.0)
            y_true = [1., -200., -300., -400.]
            @test wpicp(dist, y_true) == fill(.25, 18) # This can fail due to randomness

            α_range=0.25:0.05:0.75
            # This can fail due to randomness
            @test wpicp(dist, y_true, α_range) == fill(.25, 11)
            @test length(wpicp(dist, y_true, α_range)) == length(α_range)

            α_min=0.15
            α_max=0.85
            α_step=0.01
            # This can fail due to randomness
            @test wpicp(dist, y_true, α_min=α_min, α_max=α_max, α_step=α_step) ==
                wpicp(dist, y_true, α_min:α_step:α_max)

            y_true = [1., 3., -300., -400.]
            α_range=0.1:0.8:0.9
            # This can fail due to randomness
            @test wpicp(dist, y_true, α_range) == [.25, .5]
        end
    end

    @testset "evaluate" begin
        @testset "squared_error" begin
            dist = rand()
            y_pred = rand()
            @test evaluate(squared_error, dist, y_pred) == squared_error(dist, y_pred)
        end
    end

end
