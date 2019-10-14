@testset "regression.jl" begin

    # These tests are arranged such that they test the expected properties of a metrics.
    # Each property gets a function which evaluates the behaviour of the metric and input data.
    # Each metric then gets assigned a test_metric_properties function which contains the
    # list of relevant tests (functions) it should obey.
    # To test a new property, add a function with the appropriate tests and then add a call
    # to that function in the metrics that should obey it.

    function is_strictly_positive(metric, y_true, y_pred)
        # get mean value(s) of distribution(s)
        point_pred = mean(obs_arrangement(metric), y_pred)

        @testset "is strictly positive" begin
            @test metric(y_true, point_pred) > 0
            @test evaluate(metric, y_true, point_pred) > 0
            @test metric(y_true, y_pred) > 0
            @test evaluate(metric, y_true, y_pred) > 0
        end
    end

    function is_symmetric(metric, y_true, y_pred)
        # get mean value(s) of distribution(s)
        point_pred = mean(obs_arrangement(metric), y_pred)

        @testset "is symmetric" begin
            @test metric(y_true, point_pred) == metric(point_pred, y_true)
            @test evaluate(metric, y_true, point_pred) == evaluate(metric, point_pred, y_true)
            @test metric(y_true, y_pred) == metric(y_pred, y_true)
            @test evaluate(metric, y_true, y_pred) == evaluate(metric, y_pred, y_true)
        end
    end

    function is_not_symmetric(metric, y_true, y_pred)
        # get mean value(s) of distribution(s)
        point_pred = mean(obs_arrangement(metric), y_pred)

        # only the point prediction is tested
        @testset "is not symmetric" begin
            @test metric(y_true, point_pred) != metric(point_pred, y_true)
            @test evaluate(metric, y_true, point_pred) != evaluate(metric, point_pred, y_true)
        end
    end

    function is_zero_if_ypred_equals_ytrue(metric, y_true, y_pred)
        @testset "is zero if y_pred == y_true" begin
            @test iszero(metric(y_true, y_true))
            @test iszero(evaluate(metric, y_true, y_true))
        end
    end

    function error_increases_as_bias_increases(metric, y_true, y_pred)
        @testset "error increases as bias increases" begin

            # depending on what we are shifting we need to shift with different type
            v = if y_true isa Number
                1
            elseif eltype(y_true) <: AbstractArray
                ones.(size.(y_true))
            else
                ones(size(y_true))
            end


            y_pred0 = relocate(y_pred, y_true + 1v)
            y_pred1 = relocate(y_pred, y_true + 2v)
            y_pred2 = relocate(y_pred, y_true + 3v)

            # distribution errors
            error_d0 = metric(y_true, y_pred0)
            error_d1 = metric(y_true, y_pred1)
            error_d2 = metric(y_true, y_pred2)

            # point errors
            error_p0 = metric(y_true, mean(obs_arrangement(metric), y_pred0))
            error_p1 = metric(y_true, mean(obs_arrangement(metric), y_pred1))
            error_p2 = metric(y_true, mean(obs_arrangement(metric), y_pred2))

            @test error_d2 > error_d1 > error_d0
            @test error_p2 > error_p1 > error_p0
        end
    end

    function dist_returns_larger_errors(metric, y_true, y_pred)
        # get mean value(s) of distribution(s)
        point_pred = mean(obs_arrangement(metric), y_pred)

        # generate new distribution(s) with mean = y_true
        y_true_dist = relocate(y_pred, y_true)

        @testset "distributions return larger errors" begin
            # a distribution about the truth has metric >0
            # (whereas the true point was 0, tested above)
            @test metric(y_true, y_true_dist) > 0
            @test evaluate(metric, y_true, y_true_dist) > 0

            # distributions give larger errors for convex metrics by Jensen's inequality
            # https://en.wikipedia.org/wiki/Jensen%27s_inequality
            @test metric(y_true, y_pred) >= metric(y_true, point_pred)
            @test evaluate(metric, y_true, y_pred) >= evaluate(metric, y_true, point_pred)
        end
    end

    function dist_error_converges_safely(metric, y_true, y_pred)
        # get mean value(s) of distribution(s)
        point_pred = mean(obs_arrangement(metric), y_pred)

        # generate new distribution(s) with var = 0
        y_pred_var0 = rescale(y_pred, 0)

        @testset "distribution error converges when var = 0 " begin
            @test metric(y_true, y_pred_var0) ≈ metric(y_true, point_pred)
            @test evaluate(metric, y_true, y_pred_var0) ≈ evaluate(metric, y_true, point_pred)
        end

        @testset "does not return nan when y_pred = y_true " begin
            @test !isnan(metric(point_pred, y_pred_var0))
            @test !isnan(evaluate(metric, point_pred, y_pred_var0))
        end
    end

    function error_increases_as_var_increases(metric, y_true, y_pred)
        @testset "error increases as variance increase" begin
            y_pred0 = rescale(y_pred, 1)
            y_pred1 = rescale(y_pred, 2)
            y_pred2 = rescale(y_pred, 3)

            # distribution errors
            error_d0 = metric(y_true, y_pred0)
            error_d1 = metric(y_true, y_pred1)
            error_d2 = metric(y_true, y_pred2)

            # point errors
            error_p0 = metric(y_true, mean(obs_arrangement(metric), y_pred0))
            error_p1 = metric(y_true, mean(obs_arrangement(metric), y_pred1))
            error_p2 = metric(y_true, mean(obs_arrangement(metric), y_pred2))

            @test error_d2 > error_d1 > error_d0
            @test error_p2 == error_p1 == error_p0
        end
    end

    function errors_correctly(metric, y_true, y_pred)
        @testset "erroring" begin
            # errors if trying to compute on 2 distributions
            @test_throws MethodError metric(y_pred, y_pred)
            @test_throws MethodError evaluate(metric, y_pred, y_pred)

            # errors on dimension mismatch
            y_too_long = [y_true; y_true]
            @test_throws DimensionMismatch metric(y_too_long, y_pred)
            @test_throws DimensionMismatch evaluate(metric, y_too_long, y_pred)
        end
    end

    """expected_squared_error"""
    function test_metric_properties(metric::typeof(expected_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
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
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """mean_squared_error"""
    function test_metric_properties(metric::typeof(mean_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
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

    """mean_absolute_error"""
    function test_metric_properties(metric::typeof(mean_absolute_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """root_mean_squared_error"""
    function test_metric_properties(metric::typeof(root_mean_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """normalised_root_mean_squared_error"""
    function test_metric_properties(metric::typeof(normalised_root_mean_squared_error), args...)
        is_strictly_positive(metric, args...)
        is_not_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
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
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
        errors_correctly(metric, args...)
    end

    """mean_absolute_scaled_error"""
    function test_metric_properties(metric::typeof(mean_absolute_scaled_error), args...)
        is_strictly_positive(metric, args...)
        is_not_symmetric(metric, args...)
        is_zero_if_ypred_equals_ytrue(metric, args...)
        error_increases_as_bias_increases(metric, args...)
        dist_returns_larger_errors(metric, args...)
        dist_error_converges_safely(metric, args...)
        error_increases_as_var_increases(metric, args...)
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

        y_true_scalar = 2
        y_true_vector = [2, 3, 4]
        y_true_matrix = [1 2 3; 4 5 6]
        y_true_axis = AxisArray(y_true_vector, Axis{:obs}(["a", "b", "c"]))

        y_pred_scalar = Normal(5, 2.2)
        y_pred_vector = MvNormal([7, 6, 5], Σ)
        y_pred_matrix = MatrixNormal([1 3 5; 7 9 11], U, V)
        y_pred_index = IndexedDistribution(y_pred_vector, ["a", "b", "c"])

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

                # compute metric on indexed distribution predictions - only defined for multivariates
                if type == "vector"
                    @testset "IndexedDistribution with AbstractArray" begin
                        @test m(y_true, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "IndexedDistribution with AxisArray" begin
                        @test m(y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "IndexedDistribution with shuffled AxisArray" begin
                        names = index(y_pred_index)
                        new_order = shuffle(1:length(names))
                        _y_true_axis = AxisArray(y_true[new_order], Axis{:obs}(names[new_order]))

                        @test m(_y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, _y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                    end
                end
            end
        end

        @testset "Mean metrics on single observation" begin

            @testset "$type expected result" for (type, (y_true, y_pred)) in forecast_pairs

                y_means = mean(y_pred)

                # compute metric on point predictions
                @testset "point prediction" begin
                    @test mse(y_true, y_means) ≈ expected[typeof(se)]["point"][type] / length(y_true)
                    @test mae(y_true, y_means) ≈ expected[typeof(ae)]["point"][type] / length(y_true)
                end

                # compute metric on distribution predictions
                @testset "dist prediction" begin
                    @test mse(y_true, y_pred) ≈ expected[typeof(se)]["dist"][type] / length(y_true)
                    @test mae(y_true, y_pred) ≈ expected[typeof(ae)]["dist"][type] / length(y_true)
                end

                # compute metric on indexed distribution predictions - only defined for multivariates
                if type == "vector"
                    @testset "IndexedDistribution with AbstractArray" begin
                        @test mse(y_true, y_pred_index) ≈ expected[typeof(se)]["dist"][type] / length(y_true)
                        @test mae(y_true, y_pred_index) ≈ expected[typeof(ae)]["dist"][type] / length(y_true)
                    end
                    @testset "IndexedDistribution with AxisArray" begin
                        @test mse(y_true_axis, y_pred_index) ≈ expected[typeof(se)]["dist"][type] / length(y_true)
                        @test mae(y_true_axis, y_pred_index) ≈ expected[typeof(ae)]["dist"][type] / length(y_true)
                    end
                    @testset "IndexedDistribution with shuffled AxisArray" begin
                        names = index(y_pred_index)
                        new_order = shuffle(1:length(names))
                        _y_true_axis = AxisArray(y_true[new_order], Axis{:obs}(names[new_order]))

                        @test mse(_y_true_axis, y_pred_index) ≈ expected[typeof(se)]["dist"][type] / length(y_true)
                        @test mae(_y_true_axis, y_pred_index) ≈ expected[typeof(ae)]["dist"][type] / length(y_true)
                    end
                end
            end

        end

    end  # single obs

    @testset "collection of obs" begin
        metrics = (
            mean_squared_error,
            root_mean_squared_error,
            normalised_root_mean_squared_error,
            standardized_mean_squared_error,
            mean_absolute_error,
            mean_absolute_scaled_error,
        )

        y_true_scalar = [2, -3, 6]
        y_true_vector = [[2, 3, 0], [-9, 6, 4], [10, -2, 11]]
        y_true_matrix = [[1 2 3; 4 5 6], [4 -3 2; 1 0 -1], [2 9 0; 6 5 6]]
        y_true_axis = AxisArray.(y_true_vector, Ref(Axis{:obs}(["a", "b", "c"])))

        y_pred_scalar = Normal.([1, 0, -2], Ref(2.2))
        y_pred_vector = MvNormal.([[7, 6, 5], [-4, 0, -1], [7, 8, 5]], Ref(Σ))
        y_pred_matrix = MatrixNormal.([[1 3 5; 7 9 11], [0 0 2; 1 9 11], [-2 8 5; 7 5 3]], Ref(U), Ref(V))
        y_pred_index = IndexedDistribution.(y_pred_vector, Ref(["a", "b", "c"]))

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

                # compute metric on indexed distribution predictions - only defined for multivariates
                if type == "vector"
                    @testset "IndexedDistribution with AbstractArray" begin
                        @test m(y_true, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "IndexedDistribution with AxisArray" begin
                        @test m(y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                    end
                    @testset "IndexedDistribution with shuffled AxisArray" begin
                        # make a new vector of AxisArrays with shuffled axisnames
                        _y_true_axis = map(y_true_axis, y_pred_index) do _y_true, _y_pred
                            names = index(_y_pred)
                            new_order = shuffle(1:length(names))
                            AxisArray(_y_true[new_order], Axis{:obs}(names[new_order]))
                        end
                        @test m(_y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                        @test evaluate(m, _y_true_axis, y_pred_index) ≈ expected[typeof(m)]["dist"][type]
                    end
                end
            end
        end
    end  # collection of obs


    @testset "marginal_gaussian_loglikelihood" begin
        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_pred = [0.1, 0.2, 0.3]
            y_mean = [0.1, 0.1, 0.1]

            @test marginal_gaussian_loglikelihood(dist, y_pred) < 0.0 # logprobs always negative

            # y_pred is less likely than y_mean
            @test marginal_gaussian_loglikelihood(dist, y_pred) < marginal_gaussian_loglikelihood(dist, y_mean)

            # test arrangements
            expected = marginal_gaussian_loglikelihood(dist, y_pred)
            @test expected == evaluate(marginal_gaussian_loglikelihood, dist, y_pred)
            @test expected == evaluate(marginal_gaussian_loglikelihood, dist, Tuple(y_pred))
        end
        @testset "vector point" begin
            dist = MvNormal(3, 1.5)
            y_pred = [
                8.  10   9  11
                10   5   7  12
                10   7  10  1
            ]
            y_mean = zeros(3, 4)

            @test marginal_gaussian_loglikelihood(dist, y_pred) < 0.0 # logprobs always negative
            # y_pred is less likely than y_mean
            @test marginal_gaussian_loglikelihood(dist, y_pred) < marginal_gaussian_loglikelihood(dist, y_mean)

            # using the alternative Canonical form should not change results
            @test marginal_gaussian_loglikelihood(dist, y_pred) ≈ marginal_gaussian_loglikelihood(canonform(dist), y_pred)

            # Test observation rearragement
            expected = marginal_gaussian_loglikelihood(dist, y_pred)
            @test expected == evaluate(marginal_gaussian_loglikelihood, dist, y_pred, obsdim=2)
            obs_iter = [[8., 10, 10], [10., 5, 7], [9., 7, 10], [11., 12, 1]]
            @test expected == evaluate(marginal_gaussian_loglikelihood, dist, obs_iter)
            @test expected == evaluate(marginal_gaussian_loglikelihood, dist, y_pred'; obsdim=1)
            @test expected == evaluate(marginal_gaussian_loglikelihood, dist, y_pred')
        end
    end

    @testset "joint_gaussian_loglikelihood" begin
        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_pred = [.1, .2, .3]
            y_mean = [0.1, 0.1, 0.1]

            @test joint_gaussian_loglikelihood(dist, y_pred) < 0.0  # logprobs always negative

            # y_pred is less likely than y_mean
            @test joint_gaussian_loglikelihood(dist, y_pred) < joint_gaussian_loglikelihood(dist, y_mean)

            # For unviariate markingal and joint are the same, it is just the normalized likelyhood.
            @test joint_gaussian_loglikelihood(dist, y_pred) ≈ marginal_gaussian_loglikelihood(dist, y_pred)

            # test arrangements
            expected = joint_gaussian_loglikelihood(dist, y_pred)
            @test expected == evaluate(joint_gaussian_loglikelihood, dist, y_pred)
            @test expected == evaluate(joint_gaussian_loglikelihood, dist, Tuple(y_pred))
        end

        sqrtcov = rand(3, 3)
        @testset "vector point" for dist in (MvNormal(3, 1.5), MvNormal(zeros(3), sqrtcov*sqrtcov'))
            y_pred = [
                8.  10   9  11
                10   5   7  12
                10   7  10  1
            ]
            y_mean = zeros(3, 4)

            @test joint_gaussian_loglikelihood(dist, y_pred) < 0.0  # logprobs always negative
            # y_pred is less likely than y_mean
            @test joint_gaussian_loglikelihood(dist, y_pred) < joint_gaussian_loglikelihood(dist, y_mean)

            if dist isa ZeroMeanIsoNormal
                # For IsoNormal joint and marginal are the same, it is just the normalized likelyhood.
                @test joint_gaussian_loglikelihood(dist, y_pred) ≈ marginal_gaussian_loglikelihood(dist, y_pred)
            else
                @test joint_gaussian_loglikelihood(dist, y_pred) != marginal_gaussian_loglikelihood(dist, y_pred)
            end

            # using the alternative canonical form should not change the results
            @test joint_gaussian_loglikelihood(dist, y_pred) ≈ joint_gaussian_loglikelihood(canonform(dist), y_pred)

            # Test observation rearragement
            expected = joint_gaussian_loglikelihood(dist, y_pred)
            @test expected == evaluate(joint_gaussian_loglikelihood, dist, y_pred, obsdim=2)
            obs_iter = [[8., 10, 10], [10., 5, 7], [9., 7, 10], [11., 12, 1]]
            @test expected == evaluate(joint_gaussian_loglikelihood, dist, obs_iter)
            @test expected == evaluate(joint_gaussian_loglikelihood, dist, y_pred'; obsdim=1)
            @test expected == evaluate(joint_gaussian_loglikelihood, dist, y_pred')
        end
    end
end  # regression.jl
