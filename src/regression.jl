"""
    expected_squared_error(y_true, y_pred) -> Float64

Compute the square error between an observation `y_true` and point prediction `y_pred`.
"""
function expected_squared_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return sum(abs2, (y_true .- y_pred))
end

"""
    expected_squared_error(y_true, y_pred::Distribution) -> Float64

Compute the expected square error between an observation `y_true` and the posterior
distribution over the predicted value `y_pred`.

The expected square error of an estimator of a distribution `x` and the true value `x'` is
given by
```
ESE(x) = Var(x) + Bias(x, x')^2
```
Note: In conventional literature this function is called the "mean squared error" (MSE).
To avoid confusing this as a metric that computes the "mean" as an average over a collection
of values, we use the term "expected" to conform with the statistical nomenclature.

For more information see: https://en.wikipedia.org/wiki/Mean_squared_error#Estimator
"""
function expected_squared_error(y_true, y_pred::Distribution)
    @_dimcheck size(y_true) == size(y_pred)
    bias = mean(y_pred) - y_true
    return sum(var(y_pred)) + sum(abs2, bias)
end

function expected_squared_error(y_true, y_pred::Distribution{Matrixvariate})
    # Temporary hack
    # var and cov not yet defined for MatrixVariates so must be transformed to MultiVariate
    # can be removed when new Distribution version is tagged
    # https://github.com/JuliaStats/Distributions.jl/pull/955
    return expected_squared_error(vec(y_true), vec(y_pred))
end

expected_squared_error(y_true::Distribution, y_pred) = expected_squared_error(y_pred, y_true)
obs_arrangement(::typeof(expected_squared_error)) = SingleObs()
const se = expected_squared_error

"""
    mean_squared_error(y_true, y_pred) -> Float64

Compute the mean square error between a set of observations `y_true` and predictions `y_pred`.

"""
function mean_squared_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return mean(expected_squared_error.(y_true, y_pred))
end

obs_arrangement(::typeof(mean_squared_error)) = IteratorOfObs()
const mse = mean_squared_error

"""
    root_mean_squared_error(y_true, y_pred) -> Float64

Compute the root of the mean square error between a set of observation `y_true` and
predictions `y_pred`.
"""
root_mean_squared_error(y_true, y_pred) = √mean_squared_error(y_true, y_pred)
obs_arrangement(::typeof(root_mean_squared_error)) = IteratorOfObs()
const rmse = root_mean_squared_error

"""
    normalised_root_mean_squared_error(y_true, y_pred) -> Float64
    normalised_root_mean_squared_error(y_true, y_pred, α::Float64) -> Float64

Compute the normalised root of the mean square error between a set of observations `y_true`
and predictions `y_pred`. This is normalised by the range of `y_true` and it is scaled to
unit range. You can also normalise on the interquartile range using `α`.

https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation
"""
function normalised_root_mean_squared_error(y_true, y_pred)
    y_trues = reduce(vcat, vcat(y_true...))
    y_true_min, y_true_max = extrema(y_trues)
    return root_mean_squared_error(y_true, y_pred) / (y_true_max - y_true_min)
end

function normalised_root_mean_squared_error(y_true, y_pred, α::Float64)
    y_trues = reduce(vcat, vcat(y_true...))
    return root_mean_squared_error(y_true, y_pred) /
        (quantile(y_trues, .5 + α) - quantile(y_trues, .5 - α))
end

obs_arrangement(::typeof(normalised_root_mean_squared_error)) = IteratorOfObs()
const nrmse = normalised_root_mean_squared_error

"""
    standardized_mean_squared_error(y_true, y_pred) -> Float64

Compute the standardized mean square error between a set of observations `y_true` and
predictions `y_pred`.
"""
function standardized_mean_squared_error(y_true, y_pred)
    return mean_squared_error(y_true, y_pred) / var(norm.(y_true))
end
obs_arrangement(::typeof(standardized_mean_squared_error)) = IteratorOfObs()
const smse = standardized_mean_squared_error

"""
    expected_absolute_error(y_true, y_pred) -> Float64

Compute the total absolute error between an observation `y_true` and prediction `y_pred`.
"""
function expected_absolute_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return sum(abs.(y_true .- y_pred))
end

"""
    expected_absolute_error(y_true, y_pred::Distribution) -> Float64

Compute the expected absolute error between an observation `y_true` and the posterior
distribution over the predicted value `y_pred`.

Given a random variable `x` with mean mean `μ` and standard deviation `σ`, the expected
absolute value is described by the folded normal distribution via the following function:
```
AE(x) = μ * erf(μ / (√2 * σ)) + σ * sqrt(2/π) * exp(-μ^2 / 2σ^2)
```
where `erf` is the error function.

Note: In conventional literature this function is often called the "mean absolute error" (MAE).
To avoid confusing this as a metric that computes the "mean" as an average over a collection
of values, we use the term "expected" to conform with the statistical nomenclature.

For Multivariate and Matrixvariate distributions we compute the expected absolute error over
the individual dimensions and sum the result.

For more information see: https://en.wikipedia.org/wiki/Folded_normal_distribution
"""
function expected_absolute_error(y_true, y_pred::Distribution)
    @_dimcheck size(y_true) == size(y_pred)
    μ = mean(y_pred) - y_true
    σ = sqrt.(var(y_pred))

    mu_term = dot(μ, erf.(μ ./ (sqrt(2) * σ)))
    sigma_term = sqrt(2/π) * dot(σ, exp.(-μ.^2 ./ 2σ.^2))
    
    return mu_term + sigma_term
end


function expected_absolute_error(y_true, y_pred::Distribution{Matrixvariate})
    # Temporary hack
    # var and cov not yet defined for MatrixVariates so must be transformed to MultiVariate
    # can be removed when new Distribution version is tagged
    # https://github.com/JuliaStats/Distributions.jl/pull/955
    return expected_absolute_error(vec(y_true), vec(y_pred))
end

expected_absolute_error(y_true::Distribution, y_pred) = expected_absolute_error(y_pred, y_true)
obs_arrangement(::typeof(expected_absolute_error)) = SingleObs()
const ae = expected_absolute_error

"""
    mean_absolute_error(y_true, y_pred) -> Float64

Compute the mean absolute error between a set of observations `y_true` and point predictions `y_pred`.
"""
function mean_absolute_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    return mean(expected_absolute_error.(y_true, y_pred))
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
