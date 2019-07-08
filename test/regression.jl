@testset "regression.jl" begin

    @testset "squared_error" begin
        @testset "scalar point" begin
            y_true = 1
            y_pred = 1
            @test squared_error(y_true, y_pred) == 0
            @test evaluate(squared_error, y_true, y_pred) == 0

            y_true = 4
            @test squared_error(y_true, y_pred) == 9
            y_true = rand(Int64)
            y_pred = rand(Int64)
            @test squared_error(y_true, y_pred) == squared_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [1, 2, 3]
            y_pred = [1, 2, 3]
            @test squared_error(y_true, y_pred) == 0
            @test evaluate(squared_error, y_true, y_pred) == 0

            y_true = [2, 2, 2, 2]
            y_pred = [1, 2, 3, 4]
            @test squared_error(y_true, y_pred) == 6
            @test evaluate(squared_error, y_true, y_pred) == 6

            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test squared_error(y_true, y_pred) == squared_error(y_pred, y_true)
        end

        @testset "matrixvariate point" begin
            y_true = [
                1  2  3  4
                1  2  3  4
                1  2  3  4
            ]
            @test squared_error(y_true, y_true) == 0
            @test evaluate(squared_error, y_true, y_true) == 0

            y_pred = fill(2, 3, 4)
            @test squared_error(y_true, y_pred) == 18
            @test evaluate(squared_error, y_true, y_pred) == 18
        end

        @testset "erroring" begin
            y_true = 1
            y_pred = [1, 2, 3]
            @test_throws DimensionMismatch squared_error(y_true, y_pred)

            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch squared_error(y_true, y_pred)
            @test_throws DimensionMismatch evaluate(squared_error, y_true, y_pred) == 0
        end
    end

    @testset "mean_squared_error" begin
        @testset "normal usage" begin
            y_true = [1, 2, 3]
            @test mean_squared_error(y_true, y_true) == 0
            y_true = [2, 2, 2, 2]
            y_pred = [1, 2, 3, 4]
            @test mean_squared_error(y_true, y_pred) == 1.5
            y_true = [3, 4, 5, 6]
            @test mean_squared_error(y_true, y_pred) == 4
            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test mean_squared_error(y_true, y_pred) == mean_squared_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [
                [1,  2,  3,  4],
                [1,  2,  3,  4],
                [1,  2,  3,  4],
            ]
            @test mean_squared_error(y_true, y_true) == 0

            y_pred = fill(fill(2, 4), 3)
            @test mean_squared_error(y_true, y_pred) == 6

            y_true_m = [1 2 3 4; 1 2 3 4; 1 2 3 4]
            y_pred_m = fill(2, (3,4))
            @test evaluate(mean_squared_error, y_true_m, y_pred_m) == 6
            @test evaluate(mean_squared_error, y_true_m, y_pred_m; obsdim=1) == 6
            @test evaluate(mean_squared_error, y_true_m', y_pred_m'; obsdim=2) == 6
            y_true_nda = NamedDimsArray{(:vars, :obs)}(y_true_m')
            y_pred_nda = NamedDimsArray{(:vars, :obs)}(y_pred_m')
            @test evaluate(mean_squared_error, y_true_nda, y_pred_nda) == 6
            @test evaluate(mean_squared_error, y_true_nda', y_pred_nda') == 6
            @test evaluate(mean_squared_error, y_true_nda', y_pred_nda) == 6
            @test evaluate(mean_squared_error, y_true_nda, y_pred_nda') == 6
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  3  4],
                [1  2;  3  4],
                [1  2;  3  4],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test mean_squared_error(y_true, y_pred) == 6

            y_true_m = cat(y_true...; dims=3)
            y_pred_m = fill(2, (2, 2, 3))
            @test evaluate(mean_squared_error, y_true_m, y_pred_m; obsdim=3) == 6
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch mean_squared_error(y_true, y_pred)
        end
    end

    @testset "root_mean_squared_error" begin
        @testset "normal usage" begin
            y_true = [1, 2, 3]
            @test root_mean_squared_error(y_true, y_true) == 0
            y_true = [5, 6, 7, 8]
            y_pred = [1, 2, 3, 4]
            @test root_mean_squared_error(y_true, y_pred) == 4
            y_true = rand(1:100, 7)
            y_pred = rand(1:100, 7)
            @test root_mean_squared_error(y_true, y_pred) ==
                root_mean_squared_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [
                [1,  2,  3,  4],
                [1,  2,  3,  4],
                [1,  2,  3,  4],
            ]
            @test root_mean_squared_error(y_true, y_true) == 0
            y_true = [
                [1,  2,  4,  4],
                [1,  2,  4,  4],
                [1,  2,  4,  4],
            ]
            y_pred = fill(fill(2, 4), 3)
            @test root_mean_squared_error(y_true, y_pred) == 3
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  4  4],
                [1  2;  4  4],
                [1  2;  4  4],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test root_mean_squared_error(y_true, y_pred) == 3
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch root_mean_squared_error(y_true, y_pred)
        end
    end

    @testset "normalised_root_mean_squared_error" begin
        @testset "min and max of `y_true`" begin
            y_true = [1, 2, 3]
            @test normalised_root_mean_squared_error(y_true, y_true) == 0
            y_true = [5, 6, 7, 8, 9, 10, 11, 12, 13]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8,  9]
            @test normalised_root_mean_squared_error(y_true, y_pred) == 4/8
        end
        @testset "using quantile" begin
            y_true = [5, 6, 7, 8, 9, 10, 11, 12, 13]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8,  9]
            α = .25
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 4/4
        end
        @testset "multivariate point" begin
            y_true = [
                [1,  2,  3,  4],
                [1,  2,  3,  4],
                [1,  2,  3,  4],
            ]
            @test normalised_root_mean_squared_error(y_true, y_true) == 0
            y_true = [
                [1,  2,  4,  4],
                [1,  2,  4,  4],
                [1,  2,  4,  4],
            ]
            y_pred = fill(fill(2, 4), 3)
            @test normalised_root_mean_squared_error(y_true, y_pred) == 1
            y_true = [
                [1,  2,  4,  4],
                [2,  3,  5,  5],
                [3,  4,  6,  6],
            ]
            y_pred = [
                [2,  2,  2,  2],
                [3,  3,  3,  3],
                [4,  4,  4,  4],
            ]
            @test normalised_root_mean_squared_error(y_true, y_pred) == 0.6
            α = .5
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 0.6
            α = .25
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 1.3333333333333333
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  4  4],
                [1  2;  4  4],
                [1  2;  4  4],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test normalised_root_mean_squared_error(y_true, y_pred) == 1
            α = .5
            @test_skip normalised_root_mean_squared_error(y_true, y_pred, α) == 1
            α = .25
            @test_skip normalised_root_mean_squared_error(y_true, y_pred, α) == 1.3333333333333333
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch normalised_root_mean_squared_error(y_true, y_pred)
            α = .25
            @test_throws(
                DimensionMismatch, normalised_root_mean_squared_error(y_true, y_pred, α)
            )
        end
    end

    @testset "standardized_mean_squared_error" begin
        @testset "scalar point" begin
            y_true = [1, 2, 3]
            @test standardized_mean_squared_error(y_true, y_true) == 0
            y_true = [5, 6, 7, 8, 9, 10, 11, 12]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8]
            @test standardized_mean_squared_error(y_true, y_pred) == 16/6
        end
        @testset "multivariate point" begin
            y_true = [
                [1,  2,  3,  4],
                [2,  3,  4,  5],
                [3,  4,  5,  6],
            ]
            @test standardized_mean_squared_error(y_true, y_true) == 0
            y_true = [
                [3, -2,  0,  6],
                [2,  3,  5,  5],
                [0,  1,  3,  3],
            ]
            y_pred = fill(fill(2, 4), 3)
            @test standardized_mean_squared_error(y_true, y_pred) == 6.099189032370461
            y_true = [
                [3, -2,  0,  6],
                [2,  3,  5,  5],
                [0,  1,  3,  3],
            ]
            y_pred = [
                [2,  2,  2,  2],
                [3,  3,  3,  3],
                [4,  4,  4,  4],
            ]
            @test standardized_mean_squared_error(y_true, y_pred) == 7.067314275603867
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  5  5],
                [1  2;  4  4],
                [1  2;  3  3],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test standardized_mean_squared_error(y_true, y_pred) == 6.019086780960899
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch standardized_mean_squared_error(y_true, y_pred)
        end
    end

    @testset "absolute_error" begin
        @testset "normal usage" begin
            y_true = 1
            y_pred = 5
            @test absolute_error(y_true, y_pred) == 4
            y_true = [5]
            y_pred = [1]
            @test absolute_error(y_true, y_pred) == 4
            y_true = rand(Int64)
            y_pred = rand(Int64)
            @test absolute_error(y_true, y_pred) == absolute_error(y_pred, y_true)
            y_true = rand(Int64, 1)
            y_pred = rand(Int64, 1)
            @test absolute_error(y_true, y_pred) == absolute_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [1, 5, 2, 5, 9]
            y_pred = [5, 3, 2, 1, 1]
            @test absolute_error(y_true, y_pred) == 18
            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test absolute_error(y_true, y_pred) == absolute_error(y_pred, y_true)
        end
        @testset "matrixvariate point" begin
            y_true = [
                1  2  3  4
                1  2  3  4
                1  2  3  4
            ]
            @test absolute_error(y_true, y_true) == 0
            y_pred = fill(2, 3, 4)
            @test absolute_error(y_true, y_pred) == 12
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch absolute_error(y_true, y_pred)
        end
    end

    @testset "mean_absolute_error" begin
        @testset "normal usage" begin
            y_true = [1, 5, 2, 5, 9]
            y_pred = [5, 3, 2, 1, 1]
            @test mean_absolute_error(y_true, y_pred) == 3.6
            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test mean_absolute_error(y_true, y_pred) == mean_absolute_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [
                [1,  2,  3,  4],
                [1,  2,  3,  4],
                [1,  2,  3,  4],
            ]
            @test mean_absolute_error(y_true, y_true) == 0
            y_pred = fill(fill(2, 4), 3)
            @test mean_absolute_error(y_true, y_pred) == 4
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  3  4],
                [1  2;  3  4],
                [1  2;  3  4],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test mean_absolute_error(y_true, y_pred) == 4
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch mean_absolute_error(y_true, y_pred)
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

    @testset "evaluate" begin
        @testset "squared_error" begin
            dist = rand()
            y_pred = rand()
            @test evaluate(squared_error, dist, y_pred) == squared_error(dist, y_pred)
        end
    end

end
