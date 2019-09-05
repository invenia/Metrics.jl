@testset "regression.jl" begin

    @testset "squared_error" begin
        @testset "scalar point" begin
            y_true = 1
            y_pred = 1
            @test expected_squared_error(y_true, y_pred) == 0
            @test evaluate(expected_squared_error, y_true, y_pred) == 0

            y_true = 4
            @test expected_squared_error(y_true, y_pred) == 9
            @test evaluate(expected_squared_error, y_true, y_pred) == 9

            y_true = rand(Int64)
            y_pred = rand(Int64)
            @test expected_squared_error(y_true, y_pred) == expected_squared_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [1, 2, 3]
            y_pred = [1, 2, 3]
            @test expected_squared_error(y_true, y_pred) == 0
            @test evaluate(expected_squared_error, y_true, y_pred) == 0

            y_true = [2, 2, 2, 2]
            y_pred = [1, 2, 3, 4]
            @test expected_squared_error(y_true, y_pred) == 6
            @test evaluate(expected_squared_error, y_true, y_pred) == 6

            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test expected_squared_error(y_true, y_pred) == expected_squared_error(y_pred, y_true)
        end
        @testset "matrixvariate point" begin
            y_true = [
                1  2  3  4
                1  2  3  4
                1  2  3  4
            ]
            @test expected_squared_error(y_true, y_true) == 0
            @test evaluate(expected_squared_error, y_true, y_true) == 0

            y_pred = fill(2, 3, 4)
            @test expected_squared_error(y_true, y_pred) == 18
            @test evaluate(expected_squared_error, y_true, y_pred) == 18

        end
        @testset "erroring" begin
            y_true = 1
            y_pred = [1, 2, 3]
            @test_throws DimensionMismatch expected_squared_error(y_true, y_pred)

            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch expected_squared_error(y_true, y_pred)
            @test_throws DimensionMismatch evaluate(expected_squared_error, y_true, y_pred) == 0

            y_true = fill(1, 3, 4)
            y_pred = fill(1, 4, 4)
            @test_throws DimensionMismatch expected_squared_error(y_true, y_pred)
            @test_throws DimensionMismatch evaluate(expected_squared_error, y_true, y_pred) == 0
        end
    end

    @testset "mean_squared_error" begin
        @testset "scalar point" begin
            y_true = [1, 2, 3]
            @test mean_squared_error(y_true, y_true) == 0
            @test evaluate(mean_squared_error, y_true, y_true) == 0

            y_true = [2, 2, 2, 2]
            y_pred = [1, 2, 3, 4]
            @test mean_squared_error(y_true, y_pred) == 1.5
            @test evaluate(mean_squared_error, y_true, y_pred) == 1.5

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
            @test evaluate(mean_squared_error, y_true, y_pred) == 6

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
            @test evaluate(mean_squared_error, y_true, y_pred) == 6

            y_true_m = cat(y_true...; dims=3)
            y_pred_m = fill(2, (2, 2, 3))
            @test evaluate(mean_squared_error, y_true_m, y_pred_m; obsdim=3) == 6

        end

        @testset "univariate distribution" begin
            # multiple observations
            y_true = [2, 2, 2, 2]
            y_pred = [1, 2, 3, 4]
            dist_pred = Normal.(y_pred)
            @test mean_squared_error(y_true, dist_pred) == 2.5
            @test evaluate(mean_squared_error, y_true, dist_pred) == 2.5

            # symmetric
            @test mean_squared_error(y_true, dist_pred) == mean_squared_error(dist_pred, y_true)
        end

        @testset "multivariate distribution" begin
            # multiple observations
            y_true = fill(fill(1, 3), 3)
            y_pred = [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
            dist_pred = MvNormal.(y_pred, 1)
            @test mean_squared_error(y_true, dist_pred) == 71.0
            @test evaluate(mean_squared_error, y_true, dist_pred) == 71.0

            # symmetric
            @test mean_squared_error(y_true, dist_pred) == mean_squared_error(dist_pred, y_true)
        end

        @testset "matrixvariate distribution" begin
            # multiple observations
            y_true = [
                [1 1; 1 1],
                [2 2; 2 2],
                [3 3; 3 3],
            ]
            U = [1 2; 2 4.5]
            y_pred = fill(fill(2, 2, 2), 3)
            dist_pred = MatrixNormal.(y_pred, Ref(U), Ref(U))
            @test mean_squared_error(y_true, dist_pred) == 32.916666666666664
            @test evaluate(mean_squared_error, y_true, dist_pred) == 32.916666666666664

            # symmetric
            @test mean_squared_error(y_true, dist_pred) == mean_squared_error(dist_pred, y_true)
        end

        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch mean_squared_error(y_true, y_pred)

            y_true = [2, 2]
            dist_pred = MvNormal([1, 2, 3, 4], 1)
            @test_throws DimensionMismatch mean_squared_error(y_true, dist_pred)
        end
    end

    @testset "root_mean_squared_error" begin
        @testset "scalar point" begin
            y_true = [1, 2, 3]
            @test root_mean_squared_error(y_true, y_true) == 0

            y_true = [5, 6, 7, 8]
            y_pred = [1, 2, 3, 4]
            @test root_mean_squared_error(y_true, y_pred) == 4
            @test evaluate(root_mean_squared_error, y_true, y_pred) == 4

            y_true = rand(1:100, 7)
            y_pred = rand(1:100, 7)
            @test root_mean_squared_error(y_true, y_pred) ==
                root_mean_squared_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [
                [1,  2,  4,  4],
                [1,  2,  4,  4],
                [1,  2,  4,  4],
            ]
            @test root_mean_squared_error(y_true, y_true) == 0

            y_pred = fill(fill(2, 4), 3)
            @test root_mean_squared_error(y_true, y_pred) == 3
            @test evaluate(root_mean_squared_error, y_true, y_pred) == 3
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  4  4],
                [1  2;  4  4],
                [1  2;  4  4],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test root_mean_squared_error(y_true, y_pred) == 3
            @test evaluate(root_mean_squared_error, y_true, y_pred) == 3
        end
        @testset "univariate distribution" begin
            y_true = [5, 6, 7, 8]
            y_pred = [1, 2, 3, 4]
            dist_pred = Normal.(y_pred, Ref(1))
            @test root_mean_squared_error(y_true, dist_pred) == sqrt(17)
            @test evaluate(root_mean_squared_error, y_true, dist_pred) == sqrt(17)

            # symmetric
            @test root_mean_squared_error(y_true, dist_pred) == root_mean_squared_error(dist_pred, y_true)
        end
        @testset "multivariate distribution" begin
            # multiple observations
            y_true = [
                [1,  2,  4,  4],
                [1,  2,  4,  4],
                [1,  2,  4,  4],
            ]
            y_pred = fill(fill(2, 4), 3)
            dist_pred = MvNormal.(y_pred, Ref(2))
            @test root_mean_squared_error(y_true, dist_pred) == 5.0
            @test evaluate(root_mean_squared_error, y_true, dist_pred) == 5.0

            # symmetric
            @test root_mean_squared_error(y_true, dist_pred) == root_mean_squared_error(dist_pred, y_true)
        end
        @testset "matrixvariate distribution" begin
            # multiple observations
            y_true = [
                [1  2;  4  4],
                [1  2;  1  1],
                [2  1;  4  2],
            ]
            U = [1 2; 2 4.5]
            y_pred = fill(fill(2, 2, 2), 3)
            dist_pred = MatrixNormal.(y_pred, Ref(U), Ref(U))
            @test root_mean_squared_error(y_true, dist_pred) == 5.993051532121734
            @test evaluate(root_mean_squared_error, y_true, dist_pred) == 5.993051532121734

            # symmetric
            @test root_mean_squared_error(y_true, dist_pred) == root_mean_squared_error(dist_pred, y_true)
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch root_mean_squared_error(y_true, y_pred)

            y_true = [2, 2]
            dist_pred = MvNormal([1, 2, 3, 4], 1)
            @test_throws DimensionMismatch root_mean_squared_error(y_true, dist_pred)
        end
    end

    @testset "normalised_root_mean_squared_error" begin
        @testset "min and max of `y_true`" begin
            y_true = [1, 2, 3]
            @test normalised_root_mean_squared_error(y_true, y_true) == 0
            @test evaluate(normalised_root_mean_squared_error, y_true, y_true) == 0

            y_true = [5, 6, 7, 8, 9, 10, 11, 12, 13]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8,  9]
            @test normalised_root_mean_squared_error(y_true, y_pred) == 4/8
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred) == 4/8

        end
        @testset "using quantile" begin
            y_true = [5, 6, 7, 8, 9, 10, 11, 12, 13]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8,  9]
            α = .25
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 4/4
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred, α) == 4/4

            y_true = [5, 6, 7, 8, 9, 10, 11, 12, 13]
            dist_pred  = Normal.(y_pred)
        end
        @testset "multivariate point" begin
            y_true = [
                [1,  2,  4,  4],
                [1,  2,  4,  4],
                [1,  2,  4,  4],
            ]
            @test normalised_root_mean_squared_error(y_true, y_true) == 0
            @test evaluate(normalised_root_mean_squared_error, y_true, y_true) == 0

            y_pred = fill(fill(2, 4), 3)
            @test normalised_root_mean_squared_error(y_true, y_pred) == 1
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred) == 1

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
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred) == 0.6

            α = 0.5
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 0.6
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred, α) == 0.6

            α = 0.25
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 4/3
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred, α) == 4/3
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  4  4],
                [1  2;  4  4],
                [1  2;  4  4],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test normalised_root_mean_squared_error(y_true, y_pred) == 1
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred) == 1

            α = 0.5
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 1
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred, α) == 1

            α = 0.25
            @test normalised_root_mean_squared_error(y_true, y_pred, α) == 4/3
            @test evaluate(normalised_root_mean_squared_error, y_true, y_pred, α) == 4/3
        end
        @testset "univariate distribution" begin
            y_true = [5, 6, 7, 8, 9, 10, 11, 12, 13]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8,  9]
            dist_pred = Normal.(y_pred)
            @test normalised_root_mean_squared_error(y_true, dist_pred) == 0.5153882032022076
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred) == 0.5153882032022076

            α = 0.5
            @test normalised_root_mean_squared_error(y_true, dist_pred, α) == 0.5153882032022076
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred, α) == 0.5153882032022076
        end
        @testset "multivariate distribution" begin
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
            dist_pred = MvNormal.(y_pred, Ref(2))
            @test normalised_root_mean_squared_error(y_true, dist_pred) == 1.0
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred) == 1.0

            α = 0.5
            @test normalised_root_mean_squared_error(y_true, dist_pred, α) == 1.0
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred, α) == 1.0

            α = 0.25
            @test normalised_root_mean_squared_error(y_true, dist_pred, α) == 2.2222222222222223
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred, α) == 2.2222222222222223
        end
        @testset "matrixvariate distribution" begin
            y_true = [
                [1  2;  4  4],
                [1  2;  4  4],
                [1  2;  4  4],
            ]
            y_pred = fill(fill(2, 2, 2), 3)

            U = [1 2; 2 4.5]
            dist_pred = MatrixNormal.(y_pred, Ref(U), Ref(U))
            @test normalised_root_mean_squared_error(y_true, dist_pred) == 2.088327347690278
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred) == 2.088327347690278

            α = 0.5
            @test normalised_root_mean_squared_error(y_true, dist_pred, α) == 2.088327347690278
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred, α) == 2.088327347690278

            α = 0.25
            @test normalised_root_mean_squared_error(y_true, dist_pred, α) == 2.784436463587037
            @test evaluate(normalised_root_mean_squared_error, y_true, dist_pred, α) == 2.784436463587037
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch normalised_root_mean_squared_error(y_true, y_pred)
            α = .25
            @test_throws(
                DimensionMismatch, normalised_root_mean_squared_error(y_true, y_pred, α)
            )
            y_true = [2, 2]
            dist_pred = MvNormal(y_pred, 2)
            @test_throws DimensionMismatch normalised_root_mean_squared_error(y_true, dist_pred)
        end
    end

    @testset "standardized_mean_squared_error" begin
        @testset "scalar point" begin
            y_true = [1, 2, 3]
            @test standardized_mean_squared_error(y_true, y_true) == 0
            @test evaluate(standardized_mean_squared_error, y_true, y_true) == 0

            y_true = [5, 6, 7, 8, 9, 10, 11, 12]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8]
            @test standardized_mean_squared_error(y_true, y_pred) == 16/6
            @test evaluate(standardized_mean_squared_error, y_true, y_pred) == 16/6

        end
        @testset "multivariate point" begin
            y_true = [
                [3, -2,  0,  6],
                [2,  3,  5,  5],
                [0,  1,  3,  3],
            ]
            @test standardized_mean_squared_error(y_true, y_true) == 0
            @test evaluate(standardized_mean_squared_error, y_true, y_true) == 0

            y_pred = fill(fill(2, 4), 3)
            @test standardized_mean_squared_error(y_true, y_pred) == 6.099189032370461
            @test evaluate(standardized_mean_squared_error, y_true, y_pred) == 6.099189032370461

            y_pred = [
                [2,  2,  2,  2],
                [3,  3,  3,  3],
                [4,  4,  4,  4],
            ]
            @test standardized_mean_squared_error(y_true, y_pred) == 7.067314275603867
            @test evaluate(standardized_mean_squared_error, y_true, y_pred) == 7.067314275603867

        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  5  5],
                [1  2;  4  4],
                [1  2;  3  3],
            ]
            y_pred = fill(fill(2, 2, 2), 3)
            @test evaluate(standardized_mean_squared_error, y_true, y_pred) == 6.019086780960899
        end
        @testset "univariate distribution" begin
            y_true = [5, 6, 7, 8, 9, 10, 11, 12]
            y_pred = [1, 2, 3, 4, 5,  6,  7,  8]

            dist_pred = Normal.(y_pred)
            @test standardized_mean_squared_error(y_true, dist_pred) == 2.8333333333333335
            @test evaluate(standardized_mean_squared_error, y_true, dist_pred) == 2.8333333333333335
        end
        @testset "multivariate distribution" begin
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
            dist_pred = MvNormal.(y_pred, Ref(2))
            @test standardized_mean_squared_error(y_true, dist_pred) == 11.71431544312422
            @test evaluate(standardized_mean_squared_error, y_true, dist_pred) == 11.71431544312422
        end
        @testset "matrixvariate distribution" begin
            y_true = [
                [1  2;  5  5],
                [1  2;  4  4],
                [1  2;  3  3],
            ]
            y_pred = fill(fill(2, 2, 2), 3)

            U = [1 2; 2 4.5]
            dist_pred = MatrixNormal.(y_pred, Ref(U), Ref(U))
            @test standardized_mean_squared_error(y_true, dist_pred) == 23.63947792199966
            @test evaluate(standardized_mean_squared_error, y_true, dist_pred) == 23.63947792199966
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch standardized_mean_squared_error(y_true, y_pred)

            dist_pred = MvNormal(y_pred, 2)
            @test_throws DimensionMismatch standardized_mean_squared_error(y_true, dist_pred)
        end
    end

    @testset "absolute_error" begin
        @testset "scalar point" begin
            y_true = 1
            y_pred = 5
            @test expected_absolute_error(y_true, y_true) == 0
            @test evaluate(expected_absolute_error, y_true, y_true) == 0
            @test expected_absolute_error(y_true, y_pred) == 4
            @test evaluate(expected_absolute_error, y_true, y_pred) == 4
            @test expected_absolute_error([y_true], [y_pred]) == 4
            @test evaluate(expected_absolute_error, [y_true], [y_pred]) == 4

            y_true = rand(Int64)
            y_pred = rand(Int64)
            @test expected_absolute_error(y_true, y_pred) == expected_absolute_error(y_pred, y_true)
            @test expected_absolute_error([y_true], [y_pred]) == expected_absolute_error(y_pred, y_true)
        end
        @testset "multivariate point" begin
            y_true = [1, 5, 2, 5, 9]
            y_pred = [5, 3, 2, 1, 1]
            @test expected_absolute_error(y_true, y_pred) == 18
            @test evaluate(expected_absolute_error, y_true, y_pred) == 18

            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test expected_absolute_error(y_true, y_pred) == expected_absolute_error(y_pred, y_true)
        end
        @testset "matrixvariate point" begin
            y_true = [1  2;  5  5]
            @test expected_absolute_error(y_true, y_true) == 0
            @test evaluate(expected_absolute_error, y_true, y_true) == 0

            y_pred = fill(2, 2, 2)
            @test expected_absolute_error(y_true, y_pred) == 7
            @test evaluate(expected_absolute_error, y_true, y_pred) == 7
        end
        @testset "univariate distribution" begin
            y_true = 1
            y_pred = 5
            dist_pred = Normal(y_pred)
            @test expected_absolute_error(y_true, dist_pred) == 4.000014290516877
            @test evaluate(expected_absolute_error, y_true, dist_pred) == 4.000014290516877

            # test AE(X) = σ√(2/π) when mean(dist_pred) == y_true
            dist_pred = Normal(y_true)
            @test expected_absolute_error(y_true, dist_pred) == sqrt(2 * var(dist_pred) / π)
            @test evaluate(expected_absolute_error, y_true, dist_pred) == sqrt(2 * var(dist_pred) / π)
        end
        @testset "multivariate distribution" begin
            y_true = [1, 5, 2, 5, 9]
            y_pred = [5, 3, 2, 1, 1]
            dist_pred = MvNormal(y_pred, 1)
            @test expected_absolute_error(y_true, dist_pred) == 18.814900382239284
            @test evaluate(expected_absolute_error, y_true, dist_pred) == 18.814900382239284

            # test AE(X) = σ√(2/π) when mean(dist_pred) == y_true
            dist_pred = MvNormal(y_true, 1)
            @test expected_absolute_error(y_true, dist_pred) == sqrt(2 / π) * sum(sqrt, var(dist_pred))
            @test evaluate(expected_absolute_error, y_true, dist_pred) == sqrt(2 / π) * sum(sqrt, var(dist_pred))
        end
        @testset "matrixvariate distribution" begin
            y_true = [1  2;  5  5]
            y_pred = fill(2, 2, 2)
            U = [1 2; 2 4.5]
            dist_pred = MatrixNormal(y_pred, U, U)
            @test expected_absolute_error(y_true, dist_pred) == 10.370040141218315
            @test evaluate(expected_absolute_error, y_true, dist_pred) == 10.370040141218315

            # test AE(X) = σ√(2/π) when mean(dist_pred) == y_true
            dist_pred = MatrixNormal(y_true, U, U)
            @test expected_absolute_error(y_true, dist_pred) == sqrt(2 / π) * sum(sqrt, var(vec(dist_pred)))
            @test evaluate(expected_absolute_error, y_true, dist_pred) == sqrt(2 / π) * sum(sqrt, var(vec(dist_pred)))
        end
        @testset "erroring" begin
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch expected_absolute_error(y_true, y_pred)
        end
    end

    @testset "mean_absolute_error" begin
        @testset "scalar point" begin
            y_true = [1, 5, 2, 5, 9]
            @test mean_absolute_error(y_true, y_true) == 0
            @test evaluate(mean_absolute_error, y_true, y_true) ==0

            y_pred = [5, 3, 2, 1, 1]
            @test mean_absolute_error(y_true, y_pred) == 3.6
            @test evaluate(mean_absolute_error, y_true, y_pred) == 3.6

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
            @test evaluate(mean_absolute_error, y_true, y_true) == 0

            y_pred = fill(fill(2, 4), 3)
            @test mean_absolute_error(y_true, y_pred) == 4
            @test evaluate(mean_absolute_error, y_true, y_pred) == 4
        end
        @testset "matrixvariate point" begin
            y_true = [
                [1  2;  3  4],
                [1  2;  3  4],
                [1  2;  3  4],
            ]
            @test mean_absolute_error(y_true, y_true) == 0
            @test evaluate(mean_absolute_error, y_true, y_true) == 0

            y_pred = fill(fill(2, 2, 2), 3)
            @test mean_absolute_error(y_true, y_pred) == 4
            @test evaluate(mean_absolute_error, y_true, y_pred) == 4
        end
        @testset "univariate distribution" begin
            y_true = [1, 5, 2, 5, 9]
            y_pred = [5, 3, 2, 1, 1]
            dist_pred = Normal.(y_pred)
            @test mean_absolute_error(y_true, dist_pred) == 3.762980076447856
            @test evaluate(mean_absolute_error, y_true, dist_pred) == 3.762980076447856
        end
        @testset "multivariate distribution" begin
            y_true = [
                [1,  2,  3,  4],
                [1,  2,  3,  4],
                [1,  2,  3,  4],
            ]
            y_pred = fill(fill(2, 4), 3)
            dist_pred = MvNormal.(y_pred, Ref(2))
            @test mean_absolute_error(y_true, dist_pred) == 7.511403463166924
            @test evaluate(mean_absolute_error, y_true, dist_pred) == 7.511403463166924
        end
        @testset "matrixvariate distribution" begin
            y_true = [
                [1  2;  3  4],
                [1  2;  3  4],
                [1  2;  3  4],
            ]

            y_pred = fill(fill(2, 2, 2), 3)
            U = [1 2; 2 4.5]
            dist_pred = MatrixNormal.(y_pred, Ref(U ), Ref(U))
            @test mean_absolute_error(y_true, dist_pred) == 8.675796763888826
            @test evaluate(mean_absolute_error, y_true, dist_pred) == 8.675796763888826
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
            @test evaluate(expected_squared_error, dist, y_pred) == expected_squared_error(dist, y_pred)
        end
    end
end
