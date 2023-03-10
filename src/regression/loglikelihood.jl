"""
    marginal_gaussian_loglikelihood(y_pred::Distribution{Univariate}, y_true) -> Float64
    marginal_gaussian_loglikelihood(y_pred::Distribution{Multivariate}, y_true) -> Float64

Compute the marginal log likelihood of the data `y_true` assuming a diagonal Gaussian
distribution that is built via moment matching with the arbitrary input distribution `y_pred`.
This function takes only the diagonal elements of the covariance matrix when calculating
the probability of the points.

`marginal_gaussian_loglikelihood(y_pred, y_true) = log(P (y_true | y_pred))`
"""
function marginal_gaussian_loglikelihood(y_true, y_pred::Sampleable{Univariate})
    y_true, y_pred = _match(y_true, y_pred)
    normalized_dist = Normal(mean(y_pred), std(y_pred))
    return Distributions.loglikelihood(normalized_dist, y_true)
end

function marginal_gaussian_loglikelihood(y_true, y_pred::Sampleable{Multivariate})
    y_true, y_pred = _match(y_true, y_pred)
    # `std` is not defined on `MvNormal` so we use `sqrt.(var(...))`
    normalized_dist = MvNormal(parent(mean(y_pred)), sqrt.(parent(var(y_pred))))
    return Distributions.loglikelihood(normalized_dist, y_true)
end

function marginal_gaussian_loglikelihood(y_pred::Sampleable, y_true)
    return marginal_gaussian_loglikelihood(y_true, y_pred)
end

ObservationDims.obs_arrangement(::typeof(marginal_gaussian_loglikelihood)) = MatrixColsOfObs()

"""
    joint_gaussian_loglikelihood(y_pred::Distribution{Univariate}, y_true) -> Float64
    joint_gaussian_loglikelihood(y_pred::Distribution{Multivariate}, y_true) -> Float64

Compute the joint loglikelihood of the data `y_true` assuming a Gaussian distribution that
is built via moment matching with the arbitrary input distribution `y_pred`. This function
takes the full covariance matrix when calculating the joint probability of the points.

`joint_gaussian_loglikelihood(y_pred, y_true) = log(P (y_true | y_pred))`
"""
function joint_gaussian_loglikelihood(y_true, y_pred::Sampleable{Univariate})
    return marginal_gaussian_loglikelihood(y_true, y_pred)
end

function joint_gaussian_loglikelihood(y_true, y_pred::Sampleable{Multivariate})
    y_true, y_pred = _match(y_true, y_pred)
    normalized_dist = MvNormal(mean(y_pred), cov(y_pred))
    return Distributions.loglikelihood(normalized_dist, y_true)
end

function joint_gaussian_loglikelihood(y_pred::Sampleable, y_true)
    return joint_gaussian_loglikelihood(y_true, y_pred)
end

ObservationDims.obs_arrangement(::typeof(joint_gaussian_loglikelihood)) = MatrixColsOfObs()

"""
    loglikelihood(y_true y_pred::Sampleable{Univariate}) -> Float64
    loglikelihood(y_true::AbstractVector{<:Real}, y_pred::Sampleable{Multivariate}) -> Float64

Compute the loglikelihood of the data `y_true` under the predicted distribution `y_pred`.
Normally `y_true` is only ONE sample.
Use `Distributions.loglikelihood` under the hood with argument order swap to align with the
API in Metrics.jl


!!! note This is a seperate function from Distributions.loglikelihood

    As this follows the Metrics API which puts `y_true` first, this is a seperate function
    from `Distributions.loglikelihood`. This means if you are using both `Distributions`
    and `Metrics` you will need to specify whether your want `Distributions.loglikelihood`
    or `Metrics.loglikelihood`.
    You can do this once for a package by defining
    `const loglikelyhood = Metrics.loglikelyhood` at the start.

    Changing the argument order to [align with that is currently under discussion](https://gitlab.invenia.ca/invenia/Metrics.jl/-/issues/82).
"""
function loglikelihood(y_true, y_pred::Sampleable)
    y_true, y_pred = _match(y_true, y_pred)
    return Distributions.loglikelihood(y_pred, _maybe_parent(y_true))
end

ObservationDims.obs_arrangement(::typeof(loglikelihood)) = SingleObs()

_maybe_parent(x) = x
_maybe_parent(x::AbstractArray) = parent(x)
