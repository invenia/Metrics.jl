"""
    block_subsample(series, b)

Subsample `series` with running (overlapping) blocks of length `b`.
"""
function block_subsample(series, b)
    n_blocks = length(series) - b + 1
    return [series[i:(b + i - 1)] for i in 1:n_blocks]
end

"""
    estimate_convergence_rate(
        series,
        metric;
        quantmin=0.10,
        quantstep=0.01,
        quantmax=0.80,
        expmax=0.80,
        expstep=0.01,
        expmin=0.10,
    )

Estimate convergence rate of the estimator for `metric` over a `series`. For details on this
procedure, see chapter 8 of "Politis, Dimitris N., Joseph P. Romano, and Michael Wolf.
Subsampling. Springer Science & Business Media, 1999". An important distinction between the
approach that is suggested in the book and the one we take here is that we use the
difference between each two consecutive quantiles, instead of the quantiles themselves. That
does not affect the procedure (it's trivial to show all expressions from the book still
hold), but increases numerical stability. We also opt to use weighted linear regression
instead of regular linear regression for computing the power of the rate.

Returns the estimated exponent, `β`, assuming a rate of the form `b^β`, with `b` the size
of the block.

* Arguments:
- `series`: a numerical time-series for which to estimate convergence;
- `metric`: a function that computes the statistic for which to estimate convergence;
- `quantmin`: smallest quantile to use in the estimation;
- `quantstep`: separation between quantiles to use in the estimation;
- `quantmax`: largest quantile to use in the estimation;
- `expmax`: largest exponent to use in the estimation;
- `expstep`: separation between exponents to use in the estimation;
- `expmin`: smallest exponent to use in the estimation.
"""
# The default values here should be good for our case, which is series of the size of 1 to
# 2 years.
function estimate_convergence_rate(
    series,
    metric;
    quantmin=0.10,
    quantstep=0.01,
    quantmax=0.80,
    expmax=0.40,
    expstep=0.05,
    expmin=0.10,
)
    # Exponents used to estimate the rate.
    exps = collect(expmax:-expstep:expmin)
    n = length(series)
    block_sizes = Int.(round.(n .^ exps))

    # Get the different subsamples and apply metric to them
    subsamples = map(b -> metric.(block_subsample(series, b)), block_sizes)

    # Get the distribution of the differences of the metric
    sample_metric = metric(series)
    diff_samples = [samp .- sample_metric for samp in subsamples]

    # Define points to compute inverse cdf
    # We don't start at zero because all points must be larger than the cdf at zero
    # if st does not happen to be larger than that, the while loop will update it
    ts = collect(quantmin:quantstep:quantmax)
     if length(ts) < 3
         throw(ArgumentError(
            """
            The inverse CDF must be computed for at least 3 points for the convergence rate
            to be estimated.
            """
         ))
     end

    quant_diffs = _compute_quantile_differences(diff_samples, quantmin, quantstep, quantmax)

    # We take the negative because we want the slope for the *inverse* of the quantiles
    βw = -_compute_log_log_slope(block_sizes, quant_diffs)

    return βw
end

"""
    _compute_quantile_differences(diff_samples, quantmin, quantstep, quantmax)

Compute the differences between consecutive quantiles, ranging between `quantmin` and
`quantmax` in increments of `quantstep`, of the empirical estimate of the data `diff_samples`.
"""
function _compute_quantile_differences(diff_samples, quantmin, quantstep, quantmax)

    quant_diffs = [] # just so it lives outside of the while scope

    not_finished = true
    while not_finished
        quant_range = collect(quantmin:quantstep:quantmax)
        if length(quant_range) < 3
            throw(ArgumentError(
                """
                Can't estimate convergence rate for this series. Please provide rate
                explicitly.
                """
            ))
        end

        # Compute quantiles as the empirical estimator of the inverse cdf
        quants = [[quantile(s, t) for t in quant_range] for s in diff_samples]

        # Compute the differences between consecutive quantiles
        quant_diffs = [
            [quant[i + 1] - quant[i] for i in 1:(length(quant) - 1)] for quant in quants
        ]

        not_finished = !all([all(q .> 0) for q in quant_diffs])

        # If there are zeroes, make grid coarser
        quantstep += 0.1
    end

    return quant_diffs
end

"""
    _compute_log_log_slope(x, y::Vector{<:Vector}))

Compute the weighted least squares estimate for the slope of the line that best fits to the
natural logarithm of `x` vs the natural logarithm of `y` where `y ~ x^β`.

Every target `x_i` has multiple responses `y_{ij}`. Additionally, the weights are computed as
the inverse of the variance of the responses, i.e. `w_i` =  `1 / var.(log.(y_{ij}))`

!!! note
    This is for internal use only. The assumption made here are specific to the outputs of
    [`_compute_quantile_differences`](@ref).
"""
function _compute_log_log_slope(x, y::Vector{<:Vector})

    log_y = map(q -> mean(log.(q)), y)
    log_x = log.(x)

    diff_y = log_y .- mean(log_y)
    diff_x = log_x .- mean(log_x)

    # weight each point inversely to its variance
    weights = 1 ./ map(q -> var(log.(q)), y)

    all(isfinite.(weights)) || @warn "Not all weights are finite."

    # Use matrix inversion to compute the slope, i.e. converenge rate
    beta = sum(weights .* diff_y .* diff_x) / sum(weights .* (diff_x .^ 2))
end

"""
    estimate_block_size(
            series,
            metric;
            α=0.05,
            bmin=50,
            bmax=300,
            bstep=1,
            bvol=2,
            β=nothing,
        )

Estimate optimal block size for computing confidence intervals at a level `α` for `metric`
over a `series` by minimising their volatility. Optimal size is searched over the range
`bmin:bstep:bmax` and volatility is computed using `bvol` values above and below a given
block size. Confidence intervals are computed assuming a convergence rate of `b^β`. If
`β=nothing`, the rate is estimated via `estimate_convergence_rate`. For details on this
procedure, see chapter 9 of "Politis, Dimitris N., Joseph P. Romano, and Michael Wolf.
Subsampling. Springer Science & Business Media, 1999."

Returns the optimal block size, the (possibly estimated) `β` and the confidence interval
at level `α` with the optimal block size.
"""
function estimate_block_size(
        series,
        metric;
        α=0.05,
        bmin=50,
        bmax=300,
        bstep=1,
        bvol=2,
        β=nothing,
    )
    β = isnothing(β) ? estimate_convergence_rate(series, metric) : β
    bs = collect(bmin:bstep:bmax)
    # compute CIs for each block size
    cis = [subsample_ci(series, b, metric; α=α, β=β) for b in bs]
    # obtain lower and upper bounds
    lows = [ci[1] for ci in cis]
    ups = [ci[2] for ci in cis]
    # compute volatility indexes
    vols = [
        std(lows[i - bvol:i + bvol]) + std(ups[i - bvol:i + bvol])
        for i in (bvol + 1):(length(cis) - bvol)
    ]
    imin = findmin(vols)[2] + bvol # + bvol because length(vols) < length(cis)
    return bs[imin], β, cis[imin]
end

"""
    subsample_ci(
            series,
            metric;
            α=0.05,
            bmin=100,
            bmax=300,
            bstep=5,
            bvol=3,
            β=nothing,
        )

Compute confidence interval for `metric` over a `series` at a level `α` by estimating the
block size via `estimate_block_size`. See `estimate_block_size` for a description of the
keyword arguments.
"""
function subsample_ci(
        series,
        metric;
        α=0.05,
        bmin=50,
        bmax=300, # we may want to change this default if we use 2 years instead of 1.
        bstep=1,
        bvol=2,
        β=nothing,
    )
    return estimate_block_size(
        series,
        metric;
        α=α,
        β=β,
        bmin=bmin,
        bmax=bmax,
        bstep=bstep,
        bvol=bvol
    )[3] # we just want the CI
end

"""
    subsample_ci(series, b, metric; α=0.05, β=nothing)

Compute confidence interval for `metric` over a `series` at a level `α` using block size `b`
and convergence rate `b^β`. If `β=nothing`, the rate is estimated via
`estimate_convergence_rate`.
"""
function subsample_ci(series, b, metric; α=0.05, β=nothing)
    # apply metric to subsampled series
    metric_series = metric.(block_subsample(series, b))
    # estimate convergence rates
    β = isnothing(β) ? estimate_convergence_rate(series, metric) : β
    n = length(series)
    τ_b = b ^ β
    τ_n = n ^ β
    # compute sample metric
    sample_metric = metric(series)
    # center and scale metrics
    metric_series = (metric_series .- sample_metric) * τ_b
    # compute lower and upper bounds
    lower = quantile(metric_series, α / 2)
    upper = quantile(metric_series, 1 - (α / 2))
    # apply location and scale estimates
    lower_corrected = sample_metric - upper / τ_n
    upper_corrected = sample_metric - lower / τ_n
    return lower_corrected, upper_corrected
end
