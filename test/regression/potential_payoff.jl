@testset "potential payoff" begin

    """potential_payoff"""
    function test_metric_properties(metric::typeof(potential_payoff), args...)
        is_not_symmetric(metric, args...)
        is_nonzero_if_ypred_equals_ytrue(metric, args...)
        dist_reduces_metric_value(metric, args...)
        dist_error_converges_safely(metric, args...)
        metric_invariant_to_var(metric, args...)
        errors_correctly(metric, args...)
    end

    names = ["a", "b", "c"]

    Σ = [2 1 1; 1 2.2 2; 1 2 3]
    U = [1 2; 2 4.5]
    V = [1 2 3; 2 5.5 10.2; 3 10.2 24]

    y_true_scalar = 2
    y_true_vector = [2, 3, 4]
    y_true_matrix = [1 2 3; 4 5 6]
    y_true_keyed = KeyedArray(y_true_vector, names)

    y_pred_scalar = Normal(5, 2.2)
    y_pred_vector = MvNormal([7, 6, 5], Σ)
    y_pred_matrix = MatrixNormal([1 3 5; 7 9 11], U, V)
    y_pred_keyed = KeyedDistribution(y_pred_vector, names)


    expected = Dict(
        "dist" => Dict(
            "scalar" => 2.0,
            "vector" => 1.703703703703704,
            "matrix" => 0.31317902692489547,
        ),
        "point" => Dict(
            "scalar" => 2,
            "vector" => 2.888888888888889,
            "matrix" => 4.472222222222222,
        ),
    )

    forecast_pairs = (
        ("scalar", (y_true_scalar, y_pred_scalar)),
        ("vector", (y_true_vector, y_pred_vector)),
        ("matrix", (y_true_matrix, y_pred_matrix)),
    )
    rng = StableRNG(1)

    # test properties on all metrics and argument types
    @testset "$type properties" for (type, (y_true, y_pred)) in forecast_pairs

        test_metric_properties(potential_payoff, y_true, y_pred)

        # compute metric on KeyedDistribution predictions - only defined for multivariates
        if type == "vector"
            @testset "KeyedDistribution with AbstractArray" begin
                @test potential_payoff(y_true, y_pred_keyed) ≈ expected["dist"][type]
                @test evaluate(potential_payoff,y_true, y_pred_keyed) ≈ expected["dist"][type]
            end
            @testset "KeyedDistribution with KeyedArray" begin
                @test potential_payoff(y_true_keyed, y_pred_keyed) ≈ expected["dist"][type]
                @test evaluate(potential_payoff,y_true_keyed, y_pred_keyed) ≈ expected["dist"][type]
            end
            @testset "KeyedDistribution with shuffled KeyedArray" begin
                new_order = shuffle(rng, 1:length(names))
                _y_true_keyed = KeyedArray(y_true[new_order], obs=names[new_order])

                # Values change here because of the 'order' shuffle.
                # Instead of [1, 2, 3] we have [2, 1, 3]. This changes the 'dot' calcs in
                # potential_payoff().
                @test potential_payoff(_y_true_keyed, y_pred_keyed) ≈ 1.8518518518518523
                @test evaluate(potential_payoff,_y_true_keyed, y_pred_keyed) ≈ 1.8518518518518523
            end
        end
    end

    @testset "predicting zeros is stable" begin

        @testset "Univariate" begin

            y_pred = 0

            @test iszero(potential_payoff(y_true_scalar, y_pred))
            @test iszero(evaluate(potential_payoff, y_true_scalar, y_pred))

            @test iszero(potential_payoff(y_true_scalar, Normal(y_pred, 1.2)))
            @test iszero(evaluate(potential_payoff, y_true_scalar, Normal(y_pred, 1.2)))
        end
        @testset "Multivariate" begin

            y_pred = zeros(3)

            @test iszero(potential_payoff(y_true_vector, y_pred))
            @test iszero(evaluate(potential_payoff, y_true_vector, y_pred))

            @test iszero(potential_payoff(y_true_vector, MvNormal(y_pred, Σ)))
            @test iszero(evaluate(potential_payoff, y_true_vector, MvNormal(y_pred, Σ)))
        end
        @testset "Matrixvariate" begin

            y_pred = zeros(2, 3)

            @test iszero(potential_payoff(y_true_matrix, y_pred))
            @test iszero(evaluate(potential_payoff, y_true_matrix, y_pred))

            @test iszero(potential_payoff(y_true_matrix, MatrixNormal(y_pred, U, V)))
            @test iszero(evaluate(potential_payoff, y_true_matrix, MatrixNormal(y_pred, U, V)))
        end
    end

end
