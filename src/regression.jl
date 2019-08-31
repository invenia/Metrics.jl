"""
    squared_error(y_true, y_pred) -> Float64

Compute the total square error between a set of truths `y_true` and predictions `y_pred`.
"""
function squared_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return sum((y_true .- y_pred) .^ 2)
end

obs_arrangement(::typeof(squared_error)) = SingleObs()
const se = squared_error

se(y_true, y_pred::Distribution) = se(y_true, mean(y_pred))
se(y_true::Distribution, y_pred) = se(y_pred, y_true)

"""
    mean_squared_error(y_true, y_pred) -> Float64

Compute the mean square error between a set of truths `y_true` and predictions `y_pred`.
"""
function mean_squared_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return mean(squared_error.(y_true, y_pred))
end

obs_arrangement(::typeof(mean_squared_error)) = IteratorOfObs()
const mse = mean_squared_error


"""
    root_mean_squared_error(y_true, y_pred) -> Float64

Compute the root of the mean square error between a set of truths `y_true` and predictions
`y_pred`.
"""
root_mean_squared_error(y_true, y_pred) = √mean_squared_error(y_true, y_pred)
obs_arrangement(::typeof(root_mean_squared_error)) = IteratorOfObs()
const rmse = root_mean_squared_error

"""
    normalised_root_mean_squared_error(y_true, y_pred) -> Float64
    normalised_root_mean_squared_error(y_true, y_pred, α::Float64) -> Float64

Compute the normalised root of the mean square error between a set of truths `y_true` and
predictions `y_pred`. You can also normalised on the interquartile range using `α`. This is
normalised by the range of `y_true` and it is scaled to unit range.

https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation
"""
function normalised_root_mean_squared_error(y_true, y_pred)
    y_trues = reduce(vcat, y_true)
    y_true_min, y_true_max = extrema(y_trues)
    return root_mean_squared_error(y_true, y_pred) / (y_true_max - y_true_min)
end

function normalised_root_mean_squared_error(y_true, y_pred, α::Float64)
    y_trues = reduce(vcat, y_true)
    return root_mean_squared_error(y_true, y_pred) /
        (quantile(y_trues, .5 + α) - quantile(y_trues, .5 - α))
end

obs_arrangement(::typeof(normalised_root_mean_squared_error)) = IteratorOfObs()
const nrmse = normalised_root_mean_squared_error

"""
    standardized_mean_squared_error(y_true, y_pred) -> Float64

Compute the standardized mean square error between a set of truths `y_true` and predictions
`y_pred`.
"""
function standardized_mean_squared_error(y_true, y_pred)
    return mean_squared_error(y_true, y_pred) / var(norm.(y_true))
end
obs_arrangement(::typeof(standardized_mean_squared_error)) = IteratorOfObs()
const smse = standardized_mean_squared_error

"""
    absolute_error(y_true, y_pred) -> Float64

Compute the total absolute error between a set of truths `y_true` and predictions `y_pred`.
"""
function absolute_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return sum(abs.(y_true .- y_pred))
end

obs_arrangement(::typeof(absolute_error)) = SingleObs()
const ae = absolute_error

ae(y_true, y_pred::Distribution) = ae(y_true,  mean(y_pred))
ae(y_true::Distribution, y_pred) = ae(y_pred, y_true)

"""
    mean_absolute_error(y_true, y_pred) -> Float64

Compute the mean absolute error between a set of truths `y_true` and predictions `y_pred`.
"""
function mean_absolute_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return mean(absolute_error.(y_true, y_pred))
end
obs_arrangement(::typeof(mean_absolute_error)) = IteratorOfObs()
const mae = mean_absolute_error

"""
    marginal_loglikelihood(dist::Distribution{Univariate}, y_pred) -> Float64
    marginal_loglikelihood(dist::Distribution{Multivariate}, y_pred) -> Float64

Computes the marginal loglikelihood of the distribution `dist` given some data `y_pred`
which takes only the diagonal elements of the covariance matrix when calculating the
probability of the points.

`marginal_loglikelihood(dist, y_pred) = log(P (dist | y_pred))`
"""
function marginal_loglikelihood(dist::Distribution{Univariate}, y_pred)
    normalized_dist = Normal(mean(dist), std(dist))
    return loglikelihood(normalized_dist, y_pred)
end

function marginal_loglikelihood(dist::Distribution{Multivariate}, y_pred)
    # `std` is not defined on `MvNormal` so we use `sqrt.(var(...))`
    normalized_dist = MvNormal(mean(dist), sqrt.(var(dist)))
    return loglikelihood(normalized_dist, y_pred)
end

obs_arrangement(::typeof(marginal_loglikelihood)) = MatrixColsOfObs()


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
    normalized_dist = MvNormal(mean(dist), cov(dist))
    return loglikelihood(normalized_dist, y_pred)
end

joint_loglikelihood(dist::MvNormal, y_pred) = loglikelihood(dist, y_pred)

obs_arrangement(::typeof(joint_loglikelihood)) = MatrixColsOfObs()
