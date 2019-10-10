"""
    prediction_interval_coverage_probability(α, distribution|samples, y_trues)

Compute the Prediction Interval Coverage Probability (PICP), which is suitable for assessing
an estimated distribution against a set of true observations and is less impacted by
outliers than loglikelihoods.
A `picp` that is closer to α is bettter.
If it is below α it means the distribution is too narrow,
If it is above α it means the distribution is too wide.

PICP is bounded below at zero, which indicates that no true observations fall with in the
credible interval. It is bounded above at 1, which indicates all observations fall in the
credible interval. (To reiterate the above a score of 1 is not good)

`prediction_interval_coverage_probability` counts the portion of points that fall with-in a
credible interval of size `α`, for the distribution. The distribution may be given explictly
or via samples.

https://en.wikipedia.org/wiki/Prediction_interval
"""
function prediction_interval_coverage_probability end

# Univariate, bounds given
"""
    prediction_interval_coverage_probability(y_trues; lower_bound, upper_bound)

Lower-level univariate `picp`, where the bounds of the credible window are already known.
"""
function prediction_interval_coverage_probability(y_trues; lower_bound, upper_bound)
    return mean(lower_bound .<= y_trues .<= upper_bound)
end

# Univariate, samples
function prediction_interval_coverage_probability(α::Float64, samples::AbstractArray, y_trues)
    portions = ((1 - α)/2, (1 + α)/2)
    lower_bound, upper_bound = quantile(samples, portions)
    return prediction_interval_coverage_probability(y_trues; lower_bound=lower_bound, upper_bound=upper_bound)
end

# Univariate, distribution
function prediction_interval_coverage_probability(α::Float64, dist::UnivariateDistribution, y_trues)
    portions = ((1 - α)/2, (1 + α)/2)
    lower_bound, upper_bound = quantile.(dist, portions)
    return prediction_interval_coverage_probability(y_trues; lower_bound=lower_bound, upper_bound=upper_bound)
end

# Multivariate, gaussian
function prediction_interval_coverage_probability(α::Float64, dist::AbstractMvNormal, y_trues)
    # need to calculate how many lay in the elliptical confidence region
    # see https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval
    # See also https://github.com/JuliaStats/Distributions.jl/issues/569

    bound = quantile(Chisq(length(dist)), α)

    centroid = mean(dist)
    proj = inv(cov(dist))
    return mean(y_trues) do y
        offset = y .- centroid
        d = offset' * proj * offset
        d <= bound
    end
end

const picp = prediction_interval_coverage_probability

picp(α, dist::IndexedDistribution, y_trues) = picp(α, parent(dist), y_trues)

"""
    window_prediction_interval_coverage_probability([α_range], distribution|samples, y_trues)

Compute the @ref['picp'] over a window of quantiles, specified by `α_range` and return a
vector with all PICP values. If `α_range` is not provided this defaults to `0.1:0.05:0.95`.

This is useful either for plotting, or as a partway step for calculating @ref['apicp'].

Source: Eric P. came up with this
"""
function window_prediction_interval_coverage_probability(α_range::AbstractRange, args...)
    return [picp(α, args...) for α in α_range]
end

function window_prediction_interval_coverage_probability(args...)
    return window_prediction_interval_coverage_probability(0.1:0.05:0.95, args...)
end

function window_prediction_interval_coverage_probability(
    α_range::AbstractRange, dist::AbstractMvNormal, y_trues
)
    # This is just an optimized version of the above for the MvNormal case
    # To avoid recalculating the centroid, and proj for each α
    # It is about 20x faster than the generic fallback.

    centroid = mean(dist)
    proj = inv(cov(dist))

    ds = map(y_trues) do y
        offset = y .- centroid
        offset' * proj * offset
    end
    bound_dist = Chisq(length(dist))
    return map(α_range) do α
        bound = quantile(bound_dist, α)
        mean(ds) do d
            d<=bound
        end
    end
end

const wpicp = window_prediction_interval_coverage_probability

"""
    adjusted_prediction_interval_coverage_probability([α_range], distribution|samples, y_trues)

Compute the adjusted PICP (APICP) value over a window specified by `α_range` and return the
slope `m` of the line corresponding to the least-squares fit of @ref[`picp`] = m * α that
passes through the origin. If `α_range` is not provided this defaults to `0.1:0.05:0.95`.

This is a kind of normalizized @ref[`picp`].

A negative APICP means you have overall too much spread, while a positive APICP means you
have too little.

Source: Eric P. came up with this
"""
function adjusted_prediction_interval_coverage_probability(α_range::AbstractRange, args...)
    ps = window_prediction_interval_coverage_probability(α_range, args...)
    return dot(α_range, ps) / sum(abs2, α_range)
end

function adjusted_prediction_interval_coverage_probability(args...)
    return adjusted_prediction_interval_coverage_probability(0.1:0.05:0.95, args...)
end

const apicp = adjusted_prediction_interval_coverage_probability
