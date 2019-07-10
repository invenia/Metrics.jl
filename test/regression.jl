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
            y_true_nda2 = NamedDimsArray{(:vars, :points)}(y_true_m')
            y_pred_nda2 = NamedDimsArray{(:vars, :points)}(y_pred_m')
            @test evaluate(mean_squared_error, y_true_nda2, y_pred_nda2; obsdim=:points) == 6
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
            y_pred = [0.1, 0.2, 0.3]
            y_mean = [0.1, 0.1, 0.1]

            @test marginal_loglikelihood(dist, y_pred) < 0.0 # logprobs always negative

            # y_pred is less likely than y_mean
            @test marginal_loglikelihood(dist, y_pred) < marginal_loglikelihood(dist, y_mean)

            # test arrangements
            expected = marginal_loglikelihood(dist, y_pred)
            @test expected == evaluate(marginal_loglikelihood, dist, y_pred)
            @test expected == evaluate(marginal_loglikelihood, dist, Tuple(y_pred))
        end
        @testset "vector point" begin
            dist = MvNormal(3, 1.5)
            y_pred = [
                8.  10   9  11
                10   5   7  12
                10   7  10  1
            ]
            y_mean = zeros(3, 4)

            @test marginal_loglikelihood(dist, y_pred) < 0.0 # logprobs always negative
            # y_pred is less likely than y_mean
            @test marginal_loglikelihood(dist, y_pred) < marginal_loglikelihood(dist, y_mean)

            # using the alternative Canonical form should not change results
            @test marginal_loglikelihood(dist, y_pred) ≈ marginal_loglikelihood(canonform(dist), y_pred)

            # Test observation rearragement
            expected = marginal_loglikelihood(dist, y_pred)
            @test expected == evaluate(marginal_loglikelihood, dist, y_pred, obsdim=2)
            obs_iter = [[8., 10, 10], [10., 5, 7], [9., 7, 10], [11., 12, 1]]
            @test expected == evaluate(marginal_loglikelihood, dist, obs_iter)
            @test expected == evaluate(marginal_loglikelihood, dist, y_pred'; obsdim=1)
            @test expected == evaluate(marginal_loglikelihood, dist, y_pred')
        end
    end

    @testset "joint_loglikelihood" begin
        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_pred = [.1, .2, .3]
            y_mean = [0.1, 0.1, 0.1]

            @test joint_loglikelihood(dist, y_pred) < 0.0  # logprobs always negative

            # y_pred is less likely than y_mean
            @test joint_loglikelihood(dist, y_pred) < joint_loglikelihood(dist, y_mean)

            # For unviariate markingal and joint are the same, it is just the normalized likelyhood.
            @test joint_loglikelihood(dist, y_pred) ≈ marginal_loglikelihood(dist, y_pred)

            # test arrangements
            expected = joint_loglikelihood(dist, y_pred)
            @test expected == evaluate(joint_loglikelihood, dist, y_pred)
            @test expected == evaluate(joint_loglikelihood, dist, Tuple(y_pred))
        end

        sqrtcov = rand(3, 3)
        @testset "vector point" for dist in (MvNormal(3, 1.5), MvNormal(zeros(3), sqrtcov*sqrtcov'))
            y_pred = [
                8.  10   9  11
                10   5   7  12
                10   7  10  1
            ]
            y_mean = zeros(3, 4)

            @test joint_loglikelihood(dist, y_pred) < 0.0  # logprobs always negative
            # y_pred is less likely than y_mean
            @test joint_loglikelihood(dist, y_pred) < joint_loglikelihood(dist, y_mean)

            if dist isa ZeroMeanIsoNormal
                # For IsoNormal joint and marginal are the same, it is just the normalized likelyhood.
                @test joint_loglikelihood(dist, y_pred) ≈ marginal_loglikelihood(dist, y_pred)
            else
                @test joint_loglikelihood(dist, y_pred) != marginal_loglikelihood(dist, y_pred)
            end

            # using the alternative canonical form should not change the results
            @test joint_loglikelihood(dist, y_pred) ≈ joint_loglikelihood(canonform(dist), y_pred)


            # Test observation rearragement
            expected = joint_loglikelihood(dist, y_pred)
            @test expected == evaluate(joint_loglikelihood, dist, y_pred, obsdim=2)
            obs_iter = [[8., 10, 10], [10., 5, 7], [9., 7, 10], [11., 12, 1]]
            @test expected == evaluate(joint_loglikelihood, dist, obs_iter)
            @test expected == evaluate(joint_loglikelihood, dist, y_pred'; obsdim=1)
            @test expected == evaluate(joint_loglikelihood, dist, y_pred')
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
