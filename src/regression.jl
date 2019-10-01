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

The expected square error of an estimator of a normal distribution `X` and the true value
`X'` is given by
```math
E[X^2] = Var(X) + Bias(X, X')^2
```

Note: Jensen's inequality provides that a convex metric computed over a distribution is
greater than or equal to that of the point prediction taken from the distribution mean, thus:
```math
E[X^2] >= E[X]^2
```
See : https://en.wikipedia.org/wiki/Jensen%27s_inequality

Note: In conventional literature this function is called the "mean squared error" (MSE).
To avoid confusing this as a metric that computes the "mean" as an average over a collection
of values, we use the term "expected" to conform with the statistical nomenclature.

For Multivariate and Matrixvariate normal distributions we compute the marginal expected
squared errors over the individual dimensions and sum the result.

For more information see: https://en.wikipedia.org/wiki/Mean_squared_error#Estimator
"""
function expected_squared_error(y_true,  y_pred::Distribution)
    @_dimcheck size(y_true) == size(y_pred)
    bias = mean(y_pred) - y_true
    return sum(var(y_pred)) + sum(abs2, bias)
end

function expected_squared_error(y_true, y_pred::MatrixNormal)
    # Temporary hack
    # var and cov not yet defined for MatrixVariates so must be transformed to MultiVariate
    # can be removed when new Distribution version is tagged
    # https://github.com/JuliaStats/Distributions.jl/pull/955
    return expected_squared_error(
        vec(y_true),
        MvNormal(vec(y_pred.M), PSDMat(kron(y_pred.V.mat, y_pred.U.mat))),
    )
end

expected_squared_error(y_true::Distribution, y_pred) = expected_squared_error(y_pred, y_true)
obs_arrangement(::typeof(expected_squared_error)) = SingleObs()
const se = expected_squared_error

"""
    mean_squared_error(y_true, y_pred) -> Float64

Compute the mean square error between a set of observations `y_true` and predictions `y_pred`.

"""
function mean_squared_error(y_true::AbstractVector, y_pred::AbstractVector)
    @_dimcheck size(y_true) == size(y_pred)
    return mean(expected_squared_error.(y_true, y_pred))
end

mean_squared_error(y_true, y_pred) = sum(expected_squared_error.(y_true, y_pred)) / length(y_pred)

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
    expected_absolute_error(
        y_true, y_pred::Union{Normal, AbstractMvNormal, MatrixNormal},
    ) -> Float64

Compute the expected absolute error between an observation `y_true` and the posterior
distribution over the predicted value `y_pred`.

Given a normal random variable `X` with mean mean `μ` and standard deviation `σ`, the
expected absolute value is described by the folded normal distribution with the expected
value defined by the following function:
```math
E[|X|] = X * erf(X / (√2 * σ)) + σ * sqrt(2/π) * exp(-X^2 / 2σ^2)
```
where `erf` is the error function.

Note: Jensen's inequality provides that a convex metric computed over a distribution is
greater than or equal to that of the point prediction taken from the distribution mean, thus:
```math
E[|X|] >= |E[X]|
```
See : https://en.wikipedia.org/wiki/Jensen%27s_inequality

Note: In conventional literature this function is often called the "mean absolute error" (MAE).
To avoid confusing this as a metric that computes the "mean" as an average over a collection
of values, we use the term "expected" to conform with the statistical nomenclature.

For Multivariate and Matrixvariate normal distributions we compute the marginal expected
absolute errors over the individual dimensions and sum the result.

For more information see: https://en.wikipedia.org/wiki/Folded_normal_distribution
"""
function expected_absolute_error(y_true, y_pred::Union{Normal, AbstractMvNormal})
    @_dimcheck size(y_true) == size(y_pred)
    μ = mean(y_pred) - y_true
    σ = sqrt.(var(y_pred))
    z = μ ./ σ

    # compute the absolute error over each dimension
    abs_err = (μ .* erf.(z / √2)) + sqrt(2 / π) * (σ .* exp.(-0.5 * z.^2))

    # we can get NaNs if μ=σ=0 so we skip these when returning the result
    # this is reasonable because μ=σ=0 implies a perfect forecast in that dimension
    return all(isnan.(abs_err)) ? 0 : sum(abs_err[.!isnan.(abs_err)])

end


function expected_absolute_error(y_true, y_pred::MatrixNormal)
    # Temporary hack
    # var and cov not yet defined for MatrixVariates so must be transformed to MultiVariate
    # can be removed when new Distribution version is tagged
    # https://github.com/JuliaStats/Distributions.jl/pull/955
    return expected_absolute_error(
        vec(y_true),
        MvNormal(vec(y_pred.M), PSDMat(kron(y_pred.V.mat, y_pred.U.mat))),
    )
end

expected_absolute_error(y_true::Distribution, y_pred) = expected_absolute_error(y_pred, y_true)
obs_arrangement(::typeof(expected_absolute_error)) = SingleObs()
const ae = expected_absolute_error

"""
    mean_absolute_error(y_true, y_pred) -> Float64

Compute the mean absolute error between a set of observations `y_true` and point predictions
`y_pred`.
"""
function mean_absolute_error(y_true, y_pred::AbstractVector)
    @_dimcheck size(y_true) == size(y_pred)
    return mean(expected_absolute_error.(y_true, y_pred))
end

mean_absolute_error(y_true, y_pred) = sum(expected_absolute_error.(y_true, y_pred)) / length(y_pred)

obs_arrangement(::typeof(mean_absolute_error)) = IteratorOfObs()
const mae = mean_absolute_error

"""
    mean_absolute_scaled_error(y_true, y_pred) -> Float64

Compute the mean absolute scaled error between a set of truths `y_true` and predictions `y_pred`.

The denominator in the MASE is typically defined in the literature as the mean absolute
error of the naive one-step forecast on the model training set; that is, a model that
assigns preceding values in the training set as the forecast for the next observation.

We will instead be computing this quantity on the forecast set, i.e. the target day, (which
may be different from the training set). This makes the metric slightly less robust on
inputs with a small number of points, which may result in a denominator approximating zero.

Things to note about the implementation of MASE:
---
1. Ensure that the data is time-ordered with seasonality removed
2. Be aware that division by zero is possible and will result in `Inf`
3. MASE is non-symmetric to order of data input

https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
"""
function mean_absolute_scaled_error(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    @_dimcheck length(y_true) > 1  # One element inputs would return NaN
    model_forecast_error = mean_absolute_error(y_true, y_pred)
    one_step_forecast_error = mean_absolute_error(y_true[2:end], y_true[1:end-1])

    return model_forecast_error / one_step_forecast_error
end
obs_arrangement(::typeof(mean_absolute_scaled_error)) = IteratorOfObs()
const mase = mean_absolute_scaled_error

"""
    marginal_gaussian_loglikelihood(dist::Distribution{Univariate}, y_pred) -> Float64
    marginal_gaussian_loglikelihood(dist::Distribution{Multivariate}, y_pred) -> Float64

Compute the marginal log likelihood of the data `y_pred` assuming a Gaussian distribution
that is built via moment matching with the arbitrary input distribution `dist`. This function
takes only the diagonal elements of the covariance matrix when calculating the probability
of the points.

`marginal_gaussian_loglikelihood(dist, y_pred) = log(P (dist | y_pred))`
"""
function marginal_gaussian_loglikelihood(dist::Distribution{Univariate}, y_pred)
    normalized_dist = Normal(mean(dist), std(dist))
    return loglikelihood(normalized_dist, y_pred)
end

function marginal_gaussian_loglikelihood(dist::Distribution{Multivariate}, y_pred)
    # `std` is not defined on `MvNormal` so we use `sqrt.(var(...))`
    normalized_dist = MvNormal(mean(dist), sqrt.(var(dist)))
    return loglikelihood(normalized_dist, y_pred)
end
obs_arrangement(::typeof(marginal_gaussian_loglikelihood)) = MatrixColsOfObs()

"""
    joint_gaussian_loglikelihood(dist::Distribution{Univariate}, y_pred) -> Float64
    joint_gaussian_loglikelihood(dist::Distribution{Multivariate}, y_pred) -> Float64

Compute the joint loglikelihood of the data `y_pred` assuming a Gaussian distribution that
is built via moment matching with the arbitrary input distribution `dist`. This function
takes the full covariance matrix when calculating the joint probability of the points.

`joint_gaussian_loglikelihood(dist, y_pred) = log(P (dist | y_pred))`
"""
function joint_gaussian_loglikelihood(dist::Distribution{Univariate}, y_pred)
    return marginal_gaussian_loglikelihood(dist, y_pred)
end

function joint_gaussian_loglikelihood(dist::Distribution{Multivariate}, y_pred)
    normalized_dist = MvNormal(mean(dist), cov(dist))
    return loglikelihood(normalized_dist, y_pred)
end

joint_gaussian_loglikelihood(dist::MvNormal, y_pred) = loglikelihood(dist, y_pred)
obs_arrangement(::typeof(joint_gaussian_loglikelihood)) = MatrixColsOfObs()

"""
    regression_summary(y_true, y_pred, args...)

Calculate a summary of; @ref[`mean_squared_error`], @ref[`root_mean_squared_error`],
@ref[`normalised_root_mean_squared_error`], @ref[`standardized_mean_squared_error`],
@ref[`expected_absolute_error`], @ref[`mean_absolute_error`].

Returns a Dictionary where the `Key` is the function, and the `Value` is the result of the function.
"""
function regression_summary(args...)
    summary = Dict()

    for metric in REGRESSION_METRICS
        summary[metric] = metric(args...)
    end

    return summary
end
obs_arrangement(::typeof(regression_summary)) = IteratorOfObs()
