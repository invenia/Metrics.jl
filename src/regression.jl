"""
    evaluate(metric, grounds_truths, predictions)

Compute a metric with truths `grounds_truths` and predictions `predictions`
"""
evaluate(metric, grounds_truths, predictions) = metric(grounds_truths, predictions)

"""
    squared_error(y_true, y_pred)

Compute the total square error between a set of truths `y_true` and
predictions `y_pred`.
"""
function squared_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return sum((y_true - y_pred) .^ 2)
end

"""
    marginal_loglikelihood(dist::Distribution{Univariate}, y_pred) -> Float64
    marginal_loglikelihood(dist::Distribution{Multivariate}, y_pred) -> Float64

Computes the marginal loglikelihood of the distribution `dist` given some data `y_pred`
which takes only the diagonal elements of the covariance matrix when calculating the
probability of the points.

`marginal_loglikelihood(dist, y_pred) = log(P (dist | y_pred))`
"""
function marginal_loglikelihood(dist::Distribution{Univariate}, y_pred)
    return loglikelihood(Normal(mean(dist), std(dist)), y_pred)
end

function marginal_loglikelihood(dist::Distribution{Multivariate}, y_pred)
    # We use `.√var` instead of `std` because `std` thorws an error
    return loglikelihood(MvNormal(mean(dist), .√var(dist)), y_pred)
end

"""
    joint_loglikelihood(dist::Distribution{Univariate}, y_pred) -> Float64
    joint_loglikelihood(dist::Distribution{Multivariate}, y_pred) -> Float64

Computes the joint loglikelihood of the distribution `dist` given some data `y_pred` which
takes the full covariance matrix when calculating the joint probability of the points.

`joint_loglikelihood(dist, y_pred) = log(P (dist | y_pred))`
"""
function joint_loglikelihood(dist::Distribution{Univariate}, y_pred)
    return marginal_loglikelihood(dist, y_pred)
end

function joint_loglikelihood(dist::Distribution{Multivariate}, y_pred)
    return loglikelihood(MvNormal(mean(dist), cov(dist)), y_pred)
end
