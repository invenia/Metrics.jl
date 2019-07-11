"""
    picp(
        lower_bound::AbstractVector, upper_bound::AbstractVector, y_true::AbstractVector
    ) -> Float64
    picp(α::Float64, dist_samples::AbstractMatrix, y_true::AbstractVector) -> Float64
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
function picp end

# TODO: think about argument order

# Univariate, bounds given
function picp(y_trues; lower_bound, upper_bound)
    return mean(lower_bound .<= y_trues .<= upper_bound)
end

# Univariate, samples
function picp(α::Float64, samples, y_trues)
    portions = ((1 - α)/2, (1 + α)/2)
    lower_bound, upper_bound = quantile(samples, portions)
    return picp(y_trues; lower_bound=lower_bound, upper_bound=upper_bound)
end

# Univariate, distribution
function picp(α::Float64, dist::UnivariateDistribution, y_trues)
    portions = ((1 - α)/2, (1 + α)/2)
    lower_bound, upper_bound = quantile.(dist, portions)
    return picp(y_trues; lower_bound=lower_bound, upper_bound=upper_bound)
end

# Multivariate, gaussian
function picp(α::Float64, dist::AbstractMvNormal, y_trues)
    # need to calculate how many lay in the ellipoidal Confedence region
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


"""
    wpicp(
        dist::Distribution,
        y_true::AbstractVector,
        α_range::StepRangeLen;
        nsamples::Int=1000,
    ) -> Vector{Float64}
    wpicp(
        dist::Distribution,
        y_true::AbstractVector;
        α_min::Float64=0.10,
        α_max::Float64=0.95,
        α_step::Float64=0.05,
        nsamples::Int=1000,
    ) -> Vector{Float64}

Compute picp over a window of quantiles, specified by `α_min/max/step`. `nsamples` determines
the number of samples used for the estimation of each quantile. Returns a vector with all
`picp` values.

This is useful either for plotting, or as a partway step for calculating `apicp`.

Source: Eric P. came up with this
"""
function wpicp(
    dist::Distribution,
    y_true::AbstractVector,
    α_range::StepRangeLen;
    nsamples::Int=1000,
)
    dist_samples = rand(dist, nsamples)
    # broadcast only on `α_range`
    return picp.(α_range, Ref(dist_samples), Ref(y_true))
end
function wpicp(
    dist::Distribution,
    y_true::AbstractVector;
    α_min::Float64=0.10,
    α_max::Float64=0.95,
    α_step::Float64=0.05,
    nsamples::Int=1000,
)
    return wpicp(dist, y_true, α_min:α_step:α_max; nsamples=nsamples)
end

"""
    apicp(
        dist::Distribution,
        y_true::AbstractVector,
        α_range::StepRangeLen;
        nsamples::Int=1000,
    ) -> Float64
    apicp(
        dist::Distribution,
        y_true::AbstractVector;
        α_min::Float64=0.10,
        α_max::Float64=0.95,
        α_step::Float64=0.05,
        nsamples::Int=1000,
    ) -> Float64

Compute the adjusted `picp` value over a window specified by `α_min/max/step` and return
the slope `m` of the line corresponding to the least-squares fit of picp = m * α that
passes through the origin.

A negative picp means you have overall too much spread,
a positive picp means you have too little.

Source: Eric P. came up with this
"""
function apicp(
    dist::Distribution,
    y_true::AbstractVector,
    α_range::StepRangeLen;
    nsamples::Int=1000,
)
    ps = wpicp(dist, y_true, α_range; nsamples=nsamples)
    return dot(α_range, ps) / dot(α_range, α_range)
end
function apicp(
    dist::Distribution,
    y_true::AbstractVector;
    α_min::Float64=0.10,
    α_max::Float64=0.95,
    α_step::Float64=0.05,
    nsamples::Int=1000,
)
    return apicp(dist, y_true, α_min:α_step:α_max; nsamples=nsamples)
end
