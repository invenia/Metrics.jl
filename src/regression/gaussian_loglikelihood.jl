"""
    marginal_gaussian_loglikelihood(dist::Distribution{Univariate}, y_pred) -> Float64
    marginal_gaussian_loglikelihood(dist::Distribution{Multivariate}, y_pred) -> Float64

Compute the marginal log likelihood of the data `y_pred` assuming a Gaussian distribution
that is built via moment matching with the arbitrary input distribution `dist`. This function
takes only the diagonal elements of the covariance matrix when calculating the probability
of the points.

`marginal_gaussian_loglikelihood(dist, y_pred) = log(P (dist | y_pred))`
"""
function marginal_gaussian_loglikelihood(dist::Sampleable{Univariate}, y_pred)
    y_pred, dist = _match(y_pred, dist)
    normalized_dist = Normal(mean(dist), std(dist))
    return loglikelihood(normalized_dist, y_pred)
end

function marginal_gaussian_loglikelihood(dist::Sampleable{Multivariate}, y_pred)
    y_pred, dist = _match(y_pred, dist)
    # `std` is not defined on `MvNormal` so we use `sqrt.(var(...))`
    normalized_dist = MvNormal(parent(mean(dist)), sqrt.(parent(var(dist))))
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
function joint_gaussian_loglikelihood(dist::Sampleable{Univariate}, y_pred)
    return marginal_gaussian_loglikelihood(dist, y_pred)
end

function joint_gaussian_loglikelihood(dist::Sampleable{Multivariate}, y_pred)
    y_pred, dist = _match(y_pred, dist)
    normalized_dist = MvNormal(parent(mean(dist)), parent(cov(dist)))
    return loglikelihood(normalized_dist, y_pred)
end

joint_gaussian_loglikelihood(dist::MvNormal, y_pred) = loglikelihood(dist, y_pred)
obs_arrangement(::typeof(joint_gaussian_loglikelihood)) = MatrixColsOfObs()
