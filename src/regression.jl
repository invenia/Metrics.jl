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


"""
    picp(
        lower_bound::AbstractVector, upper_bound::AbstractVector, y_true::AbstractVector
    ) -> Float64
    picp(α::Float64, dist_samples::AbstractMatrix, y_true::AbstractVector) -> FFloat64
    picp(
        α::Float64,
        dist::Distribution{Multivariate},
        y_true::AbstractVector;
        nsamples::Int=1000,
    ) -> Float64

Prediction Interval Coverage Probability (PICP). Measure the number of measured points,
`y_true`, that fall within the credible interval defined by `lb` and `ub`.

Given a sample of the distribution `dist_samples`, find the interquantile range (`α` to
either side of the samples' median) and calculate PICP.

In order to compute this quantity, for a distribution and a given interquantile range (`α`
to either side of the median) we need the computation of the quantiles, which is done via a
Monte Carlo method, using a number of samples specified by `nsamples` (default 1000).

https://en.wikipedia.org/wiki/Prediction_interval
"""
function picp(
    lower_bound::AbstractVector, upper_bound::AbstractVector, y_true::AbstractVector
)
    return mean(lower_bound .<= y_true .<= upper_bound)
end

function picp(α::Float64, dist_samples::AbstractMatrix, y_true::AbstractVector)
    lower_bound = quantile.(eachrow(dist_samples), (1 - α)/2)
    upper_bound = quantile.(eachrow(dist_samples), (1 + α)/2)
    return picp(lower_bound, upper_bound, y_true)
end

function picp(
    α::Float64,
    dist::Distribution{Multivariate},
    y_true::AbstractVector;
    nsamples::Int=1000,
)
    return picp(α, rand(dist, nsamples), y_true)
end
