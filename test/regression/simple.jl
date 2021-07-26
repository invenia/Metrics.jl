@testset "regression.jl" begin

    """expected_squared_error"""
    function test_metric_properties(metric::typeof(expected_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)

        y_true, y_pred = args
        @testset "obeys properties of norm" begin
            # for point predictions the following holds:
            # squared_error(y, y') <= absolute_error(y, y')^2 since ||x||_2 <= ||x||_1
            @test expected_squared_error(y_true, mean(y_pred)) <=
                expected_absolute_error(y_true, mean(y_pred))^2
            @test evaluate(expected_squared_error, y_true, mean(y_pred)) <=
                evaluate(expected_absolute_error, y_true, mean(y_pred))^2
        end

        if y_pred isa Normal
            # for univariate distributions the following holds:
            # squared_error(y, y') >= absolute_error(y, y')^2 since E[X]^2 >=E[|X|]^2
            @testset "obeys properties of distributions" begin
                @test isless(
                    expected_absolute_error(y_true, y_pred)^2,
                    expected_squared_error(y_true, y_pred),
                )
                @test isless(
                    evaluate(expected_absolute_error, y_true, y_pred)^2,
                    evaluate(expected_squared_error, y_true, y_pred),
                )
            end
        end
    end

    """expected_absolute_error"""
    function test_metric_properties(metric::typeof(expected_absolute_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """mean_squared_error"""
    function test_metric_properties(metric::typeof(mean_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)

        y_true, y_pred = args
        @testset "obeys properties of norm" begin
            # for point predictions the following holds:
            # se(y, y') <= ae(y, y')^2 since ||x||_2 <= ||x||_1
            @test mse(y_true, mean.(y_pred)) <= mean(ae.(y_true, mean.(y_pred)).^2)
            @test evaluate(mse, y_true, mean.(y_pred)) <= mean(evaluate.(ae, y_true, mean.(y_pred)).^2)
        end

        if first(y_pred) isa Normal
            # for univariate distributions the following holds:
            # mean(se(y, y')) >= mean(ae(y, y')^2) since E[X]^2 >=E[|X|]^2
            @testset "obeys properties of distributions" begin
                @test mse(y_true, y_pred) >= mean(ae.(y_true, y_pred).^2)
                @test evaluate(mse, y_true, y_pred) >= mean(evaluate.(mae, y_true, y_pred).^2)
            end
        end
    end

    """mean_squared_error_to_mean"""
    function test_metric_properties(metric::typeof(mean_squared_error_to_mean), args...)
        is_strictly_positive(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        errors_correctly(metric, args...)
    end


    """mean_absolute_error"""
    function test_metric_properties(metric::typeof(mean_absolute_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """root_mean_squared_error"""
    function test_metric_properties(metric::typeof(root_mean_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """root_mean_squared_error_to_mean"""
    function test_metric_properties(metric::typeof(root_mean_squared_error_to_mean), args...)
        is_strictly_positive(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """normalised_root_mean_squared_error"""
    function test_metric_properties(metric::typeof(normalised_root_mean_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_not_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)

        @testset "equals sqrt(mse)" begin
            # rmse == sqrt(mse)
            @test rmse(args...) == √mse(args...)
            @test evaluate(rmse, args...) == √evaluate(mse, args...)
        end
    end

    """standardized_root_mean_squared_error"""
    function test_metric_properties(metric::typeof(standardized_mean_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_not_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """mean_absolute_scaled_error"""
    function test_metric_properties(metric::typeof(mean_absolute_scaled_error), args...)
        is_strictly_positive(metric, args...)
        is_not_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_increases_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)

        @testset "SpiderFinancial Example" begin
            # Example taken from: https://tinyurl.com/y35p8j63
            y_true = [
                -2.9, -2.83, -0.95, -0.88, 1.21, -1.67, 0.83, -0.27, 1.36, -0.34,
                0.48, -2.83, -0.95, -0.88, 1.21, -1.67, -2.99, 1.24, 0.64,
            ]
            y_pred = [
                -2.95, -2.7, -1, -0.68, 1.5, -1, 0.9, -0.37, 1.26, -0.54, 0.58,
                -2.13, -0.75, -0.89, 1.25, -1.65, -3.20, 1.29, 0.6,
            ]

            expected = 0.09832904884318767

            @test mase(y_true, y_pred) == expected
            @test evaluate(mase, y_true, y_pred) == expected
        end
    end

    # constants for defining Distributions
    Σ = [2 1 1; 1 2.2 2; 1 2 3]
    U = [1 2; 2 4.5]
    V = [1 2 3; 2 5.5 10.2; 3 10.2 24]

    @testset "single obs" begin
        metrics = (expected_squared_error, expected_absolute_error)

        names = ["a", "b", "c"]

        y_true_scalar = 2
        y_true_vector = [2, 3, 4]
        y_true_matrix = [1 2 3; 4 5 6]
        y_true_keyed = KeyedArray(y_true_vector, obs=names)

        y_pred_scalar = Normal(5, 2.2)
        y_pred_vector = MvNormal([7, 6, 5], Σ)
        y_pred_matrix = MatrixNormal([1 3 5; 7 9 11], U, V)
        y_pred_keyed = KeyedDistribution(y_pred_vector, names)

        expected = Dict(
            typeof(expected_squared_error) => Dict(
                "dist" => Dict(
                    "scalar" => 9 + 2.2^2,
                    "vector" => 35 + (2 + 2.2 + 3),
                    "matrix" => 55 + 167.75,
                ),
                "point" => Dict(
                    "scalar" => 9,
                    "vector" => 35,
                    "matrix" => 55,
                )
            ),
            typeof(expected_absolute_error) => Dict(
                "dist" => Dict(
                    "scalar" => 3.174703901266667,
                    "vector" => 9.62996118474158,
                    "matrix" => 24.638584200445532,
                ),
                "point" => Dict(
                    "scalar" => 3,
                    "vector" => 9,
                    "matrix" => 15,
                )
            ),
        )

        forecast_pairs = (
            ("scalar", (y_true_scalar, y_pred_scalar)),
            ("vector", (y_true_vector, y_pred_vector)),
            ("matrix", (y_true_matrix, y_pred_matrix)),
        )

        @testset "$m" for m in metrics

            # test properties on all metrics and argument types
            @testset "$type properties" for (type, args) in forecast_pairs
                test_metric_properties(m, args...)
            end

            # test expected results on all metrics and argument types
            @testset "$type expected result" for (type, (y_true, y_pred)) in forecast_pairs

                y_means = mean(y_pred)

                # compute metric on point predictions
                @testset "point prediction" begin
                    @test m(y_true, y_means) ≈ expected[typeof(m)]["point"][type]
                    @test evaluate(m, y_true, y_means) ≈ expected[typeof(m)]["point"][type]
                end

                # compute metric on distribution predictions
                @testset "dist prediction" begin
                    @test m(y_true, y_pred) ≈ expected[typeof(m)]["dist"][type]
                    @test evaluate(m, y_true, y_pred) ≈ expected[typeof(m)]["dist"][type]
                end

                # compute metric on KeyedDistribution predictions - only defined for multivariates
                if type == "vector"
                    @testset "KeyedDistribution with AbstractArray" begin
                        @test m(y_true, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "KeyedDistribution with KeyedArray" begin
                        @test m(y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "KeyedDistribution with shuffled KeyedArray" begin
                        new_order = shuffle(1:length(names))
                        _y_true_keyed = KeyedArray(y_true[new_order], obs=names[new_order])

                        @test m(_y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, _y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                    end
                end
            end
        end

        @testset "Mean metrics on single observation" begin

            @testset "$type expected result" for (type, (y_true, y_pred)) in forecast_pairs

                @testset "$mean_metric" for (mean_metric, single_metric) in ((mse, se), (mae, ae))

                    y_means = mean(y_pred)

                    # compute metric on point predictions
                    @testset "point prediction" begin
                        @test isapprox(
                            mean_metric(y_true, y_means),
                            expected[typeof(single_metric)]["point"][type] / length(y_true)
                        )
                    end

                    # compute metric on distribution predictions
                    @testset "dist prediction" begin
                        @test isapprox(
                            mean_metric(y_true, y_pred),
                            expected[typeof(single_metric)]["dist"][type] / length(y_true)
                        )
                    end

                    # compute metric on KeyedDistribution predictions - only defined for multivariates
                    if type == "vector"
                        @testset "KeyedArray with KeyedArray" begin
                            y_pred_keyarr = KeyedArray(y_means, obs=names)

                            @test isapprox(
                                mean_metric(y_true_keyed, y_pred_keyarr),
                                expected[typeof(single_metric)]["point"][type] / length(y_true)
                            )
                        end
                        @testset "KeyedArray with shuffled KeyedArray" begin
                            new_order = shuffle(1:length(names))
                            y_pred_keyarr = KeyedArray(y_means, obs=names[new_order])

                            @test isapprox(
                                mean_metric(y_true_keyed, y_pred_keyarr),
                                expected[typeof(single_metric)]["point"][type] / length(y_true)
                            )
                        end
                        @testset "KeyedArrays don't match" begin
                            y_pred_keyarr = KeyedArray(y_means, obs=["a", "b", "q"])
                            @test_throws ArgumentError mean_metric(y_true_keyed, y_pred_keyarr)
                        end
                        @testset "KeyedDistribution with AbstractArray" begin
                            @test isapprox(
                                mean_metric(y_true, y_pred_keyed),
                                expected[typeof(single_metric)]["dist"][type] / length(y_true)
                            )
                        end
                        @testset "KeyedDistribution with KeyedArray" begin
                            @test isapprox(
                                mean_metric(y_true_keyed, y_pred_keyed),
                                expected[typeof(single_metric)]["dist"][type] / length(y_true)
                            )
                        end
                        @testset "KeyedDistribution with shuffled KeyedArray" begin
                            new_order = shuffle(1:length(names))
                            _y_true_keyed = KeyedArray(y_true[new_order], obs=names[new_order])

                            @test isapprox(
                                mean_metric(_y_true_keyed, y_pred_keyed),
                                expected[typeof(single_metric)]["dist"][type] / length(y_true)
                            )
                        end
                        @testset "KeyedDistribution and KeyedArray don't match" begin
                            _y_true_keyed = KeyedArray(y_true, obs=["a", "z", "t"])
                            @test_throws ArgumentError mean_metric(_y_true_keyed, y_pred_keyed)
                        end
                    end  # if vector
                end # mean metrics
            end # type
        end
    end  # single obs

    @testset "collection of obs" begin
        metrics = (
            mean_squared_error,
            mean_squared_error_to_mean,
            root_mean_squared_error,
            root_mean_squared_error_to_mean,
            normalised_root_mean_squared_error,
            standardized_mean_squared_error,
            mean_absolute_error,
            mean_absolute_scaled_error,
        )

        names = ["a", "b", "c"]

        y_true_scalar = [2, -3, 6]
        y_true_vector = [[2, 3, 0], [-9, 6, 4], [10, -2, 11]]
        y_true_matrix = [[1 2 3; 4 5 6], [4 -3 2; 1 0 -1], [2 9 0; 6 5 6]]
        y_true_keyed = [KeyedArray(y; obs=names) for y in y_true_vector]

        y_pred_scalar = Normal.([1, 0, -2], Ref(2.2))
        y_pred_vector = MvNormal.([[7, 6, 5], [-4, 0, -1], [7, 8, 5]], Ref(Σ))
        y_pred_matrix = MatrixNormal.([[1 3 5; 7 9 11], [0 0 2; 1 9 11], [-2 8 5; 7 5 3]], Ref(U), Ref(V))
        y_pred_keyed = KeyedDistribution.(y_pred_vector, Ref(names))

        expected = Dict(
            typeof(mean_squared_error) => Dict(
                "dist" => Dict(
                    "scalar" => 74 / 3 + 2.2^2,
                    "vector" => (290 / 3 + (2 + 2.2 + 3)) / 3,
                    "matrix" => (357 / 3 + 167.75) / 6,
                ),
                "point" => Dict(
                    "scalar" => 74 / 3,
                    "vector" => 290 / 9,
                    "matrix" => 357 / 18,
                )
            ),
            typeof(mean_squared_error_to_mean) => Dict(
                "dist" => Dict(
                    "scalar" => 74 / 3,
                    "vector" => (290 / 3) / 3,
                    "matrix" => (357 / 3) / 6,
                ),
                "point" => Dict(
                    "scalar" => 74 / 3,
                    "vector" => 290 / 9,
                    "matrix" => 357 / 18,
                )
            ),
            typeof(root_mean_squared_error) => Dict(
                "dist" => Dict(
                    "scalar" => sqrt(74 / 3 + 2.2^2),
                    "vector" => sqrt((290 / 3 + (2 + 2.2 + 3)) / 3),
                    "matrix" => sqrt((357 / 3 + 167.75) / 6),
                ),
                "point" => Dict(
                    "scalar" => sqrt(74 / 3),
                    "vector" => sqrt(290 / 9),
                    "matrix" => sqrt(357 / 18),
                )
            ),
            typeof(root_mean_squared_error_to_mean) => Dict(
                "dist" => Dict(
                    "scalar" => sqrt(74 / 3),
                    "vector" => sqrt((290 / 3) / 3),
                    "matrix" => sqrt((357 / 3) / 6),
                ),
                "point" => Dict(
                    "scalar" => sqrt(74 / 3),
                    "vector" => sqrt(290 / 9),
                    "matrix" => sqrt(357 / 18),
                )
            ),
            typeof(normalised_root_mean_squared_error) => Dict(
                "dist" => Dict(
                    "scalar" => sqrt(74 / 3 + 2.2^2) / 9,
                    "vector" => sqrt((290 / 3 + (2 + 2.2 + 3)) / 3) / 20,
                    "matrix" => sqrt((357 / 3 + 167.75) / 6) / 12,
                ),
                "point" => Dict(
                    "scalar" => sqrt(74 / 3) / 9,
                    "vector" => sqrt(290 / 9) / 20,
                    "matrix" => sqrt(357 / 18) / 12,
                )
            ),
            typeof(standardized_mean_squared_error) => Dict(
                "dist" => Dict(
                    "scalar" => (74 / 3 + 2.2^2) / 4.333333333333334,
                    "vector" => ((290 / 3 + (2 + 2.2 + 3)) / 3) / 34.115682058464884,
                    "matrix" => ((357 / 3 + 167.75) / 6) / 15.693410363878222,
                ),
                "point" => Dict(
                    "scalar" => (74 / 3) / 4.333333333333334,
                    "vector" => (290 / 9) / 34.115682058464884,
                    "matrix" => (357 / 18) / 15.693410363878222,
                )
            ),
            typeof(mean_absolute_error) => Dict(
                "dist" => Dict(
                    "scalar" => 4.369492269221085,
                    "vector" => 16.015137998532605 / 3,
                    "matrix" => 28.728598682194946 / 6,
                ),
                "point" => Dict(
                    "scalar" => 12 / 3,
                    "vector" => 48 / 9,
                    "matrix" => 57 / 18,
                ),
            ),
            typeof(mean_absolute_scaled_error) => Dict(
                "dist" => Dict(
                    "scalar" => 4.369492269221085 / 7,
                    "vector" => 16.015137998532605 / 26.0,
                    "matrix" => 28.728598682194946 / 28.5,
                ),
                "point" => Dict(
                    "scalar" => 12 / 21,
                    "vector" => 48 / 78,
                    "matrix" => 57 / 85.5,
                ),
            )
        )

        forecast_pairs = (
            ("scalar", (y_true_scalar, y_pred_scalar)),
            ("vector", (y_true_vector, y_pred_vector)),
            ("matrix", (y_true_matrix, y_pred_matrix)),
        )

        @testset "$m" for m in metrics

            # test properties on all metrics and argument types
            @testset "$type properties" for (type, args) in forecast_pairs
                test_metric_properties(m, args...)
            end

            # test expected results on all metrics and argument types
            @testset "$type expected result" for (type, (y_true, y_pred)) in forecast_pairs

                y_means = mean.(y_pred)

                # compute metric on point predictions
                @testset "point prediction" begin
                    @test m(y_true, y_means) ≈ expected[typeof(m)]["point"][type]
                    @test evaluate(m, y_true, y_means) ≈ expected[typeof(m)]["point"][type]
                end

                # compute metric on distribution predictions
                @testset "dist prediction" begin
                    @test m(y_true, y_pred) ≈ expected[typeof(m)]["dist"][type]
                    @test evaluate(m, y_true, y_pred) ≈ expected[typeof(m)]["dist"][type]
                end

                # compute metric on KeyedDistribution predictions - only defined for multivariates
                if type == "vector"
                    @testset "KeyedArray with KeyedArray" begin
                        # make a new vector of predicted KeyedArrays
                        y_pred_keyarr = [KeyedArray(y, obs=names) for y in y_means]
                        @test m(y_true_keyed, y_pred_keyarr) ≈ expected[typeof(m)]["point"][type]
                    end
                    @testset "KeyedArray with shuffled KeyedArray" begin
                        # make a new vector of predicted KeyedArrays with shuffles axis names
                        new_order = shuffle(1:length(names))
                        y_pred_keyarr = [KeyedArray(y[new_order], obs=names[new_order]) for y in y_means]
                        @test m(y_true_keyed, y_pred_keyarr) ≈ expected[typeof(m)]["point"][type]
                    end
                    @testset "KeyedArrays don't match" begin
                        # make a new vector of predicted KeyedArrays with mismatched axis names
                        y_pred_keyarr = [KeyedArray(y, obs=["b", "c", "q"]) for y in y_means]
                        @test_throws ArgumentError m(y_true_keyed, y_pred_keyarr)
                    end
                    @testset "KeyedDistribution with AbstractArray" begin
                        @test m(y_true, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "KeyedDistribution with KeyedArray" begin
                        @test m(y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "KeyedDistribution with shuffled KeyedArray" begin
                        # make a new vector of KeyedArrays with shuffled dimnames
                        new_order = shuffle(1:length(names))
                        _y_true_keyed = [y[new_order] for y in y_true_keyed]

                        @test m(_y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, _y_true_keyed, y_pred_keyed) ≈ expected[typeof(m)]["dist"][type]
                    end
                end
            end  # expected results
        end  # metric
    end  # collection of obs
end  # regression.jl
