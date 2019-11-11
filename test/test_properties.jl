# These tests are arranged such that they test the expected properties of a metrics.
# Each property gets a function which evaluates the behaviour of the metric and input data.
# Each metric then gets assigned a test_metric_properties function which contains the
# list of relevant tests (functions) it should obey.
# To test a new property, add a function with the appropriate tests and then add a call
# to that function in the metrics that should obey it.

function is_strictly_positive(metric, y_true, y_pred)
    # get mean value(s) of distribution(s)
    point_pred = mean(y_pred)

    @testset "is strictly positive" begin
        @test metric(y_true, point_pred) > 0
        @test evaluate(metric, y_true, point_pred) > 0
        @test metric(y_true, y_pred) > 0
        @test evaluate(metric, y_true, y_pred) > 0
    end
end

function is_symmetric(metric, y_true, y_pred)
    # get mean value(s) of distribution(s)
    point_pred = mean(y_pred)

    @testset "is symmetric" begin
        @test metric(y_true, point_pred) == metric(point_pred, y_true)
        @test evaluate(metric, y_true, point_pred) == evaluate(metric, point_pred, y_true)
        @test metric(y_true, y_pred) == metric(y_pred, y_true)
        @test evaluate(metric, y_true, y_pred) == evaluate(metric, y_pred, y_true)
    end
end

function is_not_symmetric(metric, y_true, y_pred)
    # get mean value(s) of distribution(s)
    point_pred = mean(y_pred)

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

function is_nonzero_if_ypred_equals_ytrue(metric, y_true, y_pred)
    @testset "is non-zero if y_pred == y_true" begin
        @test !iszero(metric(y_true, y_true))
        @test !iszero(evaluate(metric, y_true, y_true))
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
        error_p0 = metric(y_true, mean(y_pred0))
        error_p1 = metric(y_true, mean(y_pred1))
        error_p2 = metric(y_true, mean(y_pred2))

        @test error_d2 > error_d1 > error_d0
        @test error_p2 > error_p1 > error_p0
    end
end

function dist_increases_metric_value(metric, y_true, y_pred)
    # get mean value(s) of distribution(s)
    point_pred = mean(y_pred)

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

function dist_reduces_metric_value(metric, y_true, y_pred)
    # get mean value(s) of distribution(s)
    point_pred = mean(y_pred)

    # generate new distribution(s) with mean = y_true
    y_true_dist = relocate(y_pred, y_true)

    @testset "distributions return smaller errors" begin
        @test metric(y_true, y_true_dist) > 0
        @test evaluate(metric, y_true, y_true_dist) > 0

        @test metric(y_true, y_pred) <= metric(y_true, point_pred)
        @test evaluate(metric, y_true, y_pred) <= evaluate(metric, y_true, point_pred)
    end
end

function dist_error_converges_safely(metric, y_true, y_pred)
    # get mean value(s) of distribution(s)
    point_pred = mean(y_pred)

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

function dist_error_converges_safely(metric::typeof(potential_payoff), y_true, y_pred)
    # get mean value(s) of distribution(s)
    point_pred = mean(y_pred)

    # generate new distribution(s) with var = 1
    y_pred_var1 = unit_variance(y_pred)

    @testset "distribution error converges when var = 1 " begin
        @test metric(y_true, y_pred_var1) ≈ metric(y_true, point_pred)
        @test evaluate(metric, y_true, y_pred_var1) ≈ evaluate(metric, y_true, point_pred)
    end

    @testset "does not return nan when y_pred = y_true " begin
        @test !isnan(metric(point_pred, y_pred_var1))
        @test !isnan(evaluate(metric, point_pred, y_pred_var1))
    end
end

function metric_increases_as_var_increases(metric, y_true, y_pred)
    @testset "metric increases as variance increase" begin
        y_pred0 = rescale(y_pred, 1)
        y_pred1 = rescale(y_pred, 2)
        y_pred2 = rescale(y_pred, 3)

        # distribution errors
        error_d0 = metric(y_true, y_pred0)
        error_d1 = metric(y_true, y_pred1)
        error_d2 = metric(y_true, y_pred2)

        # point errors
        error_p0 = metric(y_true, mean(y_pred0))
        error_p1 = metric(y_true, mean(y_pred1))
        error_p2 = metric(y_true, mean(y_pred2))

        @test error_d2 > error_d1 > error_d0
        @test error_p2 == error_p1 == error_p0
    end
end

function metric_decreases_as_var_increases(metric, y_true, y_pred)
    @testset "metric decreases as variance increase" begin
        y_pred0 = rescale(y_pred, 1)
        y_pred1 = rescale(y_pred, 2)
        y_pred2 = rescale(y_pred, 3)

        # distribution errors
        error_d0 = metric(y_true, y_pred0)
        error_d1 = metric(y_true, y_pred1)
        error_d2 = metric(y_true, y_pred2)

        # point errors
        error_p0 = metric(y_true, mean(y_pred0))
        error_p1 = metric(y_true, mean(y_pred1))
        error_p2 = metric(y_true, mean(y_pred2))

        @test error_d2 < error_d1 < error_d0
        @test error_p2 == error_p1 == error_p0
    end
end

function metric_invariant_to_var(metric, y_true, y_pred)
    @testset "metric is invariant to variance" begin
        y_pred0 = rescale(y_pred, 1)
        y_pred1 = rescale(y_pred, 2)
        y_pred2 = rescale(y_pred, 3)

        # distribution errors
        error_d0 = metric(y_true, y_pred0)
        error_d1 = metric(y_true, y_pred1)
        error_d2 = metric(y_true, y_pred2)

        # point errors
        error_p0 = metric(y_true, mean(y_pred0))
        error_p1 = metric(y_true, mean(y_pred1))
        error_p2 = metric(y_true, mean(y_pred2))

        @test error_d2 ≈ error_d1 ≈ error_d0
        @test error_p2 ≈ error_p1 ≈ error_p0
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
