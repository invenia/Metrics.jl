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
    bs = Int.(round.(n .^ exps)) # Block sizes
    # Get the different subsamples and apply metric to them
    subsamples = map(b -> metric.(block_subsample(series, b)), bs)
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

    proceed = false
    quants = [] # just so it lives outside of the while scope
    quant_diffs = [] # just so it lives outside of the while scope
    while !proceed
        # Compute quantiles as the empirical estimator of the inverse cdf
        quants = [[quantile(s, t) for t in ts] for s in diff_samples]
        # Compute the differences between consecutive quantiles
        quant_diffs = [
            [quant[i + 1] - quant[i] for i in 1:(length(quant) - 1)] for quant in quants
        ]

        if all([prod(q .> 0) for q in quant_diffs]) # No zeroes
            proceed = true
        else
            quantstep += 0.1
            ts = collect(quantmin:quantstep:quantmax)
            if length(ts) < 3
                throw(ArgumentError(
                    """
                    Can't estimate convergence rate for this series. Please provide rate
                    explicitly.
                    """
                ))
            end
        end
    end
    ys = [mean(log.(q)) for q in quant_diffs]
    ws = 1 ./ [var(log.(q)) for q in quant_diffs]
    ȳ = mean(ys)
    mlog = mean(log.(bs))
    diff_y = ys .- ȳ
    diff_log = log.(bs) .- mlog
    # Perform weighted linear regression
    βw = - sum(ws .* diff_y .* diff_log) / sum(ws .* (diff_log .^ 2))
    return βw
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
    τ = b ^ β
    # compute sample metric
    sample_metric = metric(series)
    # center and scale metrics
    metric_series = (metric_series .- sample_metric) * τ
    # compute lower and upper bounds
    lower = quantile(metric_series, α / 2)
    upper = quantile(metric_series, 1 - (α / 2))
    # apply location and scale estimates
    lower_corrected = sample_metric - upper / τ
    upper_corrected = sample_metric - lower / τ
    return lower_corrected, upper_corrected
end
