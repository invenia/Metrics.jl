"""
    block_subsample(series, block_size; circular=false)

Subsample `series` with running (overlapping) blocks of length `b`.

If `circular`, blocks wrap around the end of the `series`.
"""
function block_subsample(series, block_size; circular=false)

    if block_size > length(series)
        throw(DomainError(
            block_size,
            "block_size cannot exceed the series length $(length(series))."
        ))
    end
    n_blocks = length(series) - block_size + 1
    regular = [series[i:(block_size + i - 1)] for i in 1:n_blocks]
    if !circular
        return regular
    else
        wrapping = [
            vcat(series[(end - block_size + i):end], series[1:i - 1]) for i in 2:block_size
        ]
        return vcat(regular, wrapping)
    end
end

"""
    estimate_convergence_rate(
        metric, series;
        quantmin=0.10, quantstep=0.01, quantmax=0.80, expmax=0.40, expstep=0.05, expmin=0.10,
    )

Estimate the convergence rate of the estimator for `metric` over a `series`. For details on
this procedure, see chapter 8 of "Politis, Dimitris N., Joseph P. Romano, and Michael Wolf.
Subsampling. Springer Science & Business Media, 1999".

An important distinction between the approach that is suggested in the book and the one we
take here is that we use the _difference_ between each two consecutive quantiles, instead of
the quantiles themselves. That does not affect the procedure, as it's trivial to show all
expressions from the book still hold, but increases numerical stability. We also opt to use
weighted linear regression instead of regular linear regression for computing the power of
the rate.

Returns the estimated exponent, `β`, assuming a rate of the form `b^β`, with `b` the size
of the block.

The default values for the quantile and exponent ranges should be good for our use case,
which is a series of length 1 to 2 years.

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
function estimate_convergence_rate(
    metric::Function, series;
    quantmin=0.10, quantstep=0.01, quantmax=0.80, expmax=0.40, expstep=0.05, expmin=0.10,
)
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

    # Exponents used to estimate the rate.
    exps = collect(expmax:-expstep:expmin)
    n = length(series)
    block_sizes = Int.(round.(n .^ exps))

    # Get the different subsamples and apply metric to them
    subsamples = map(b -> metric.(block_subsample(series, b)), block_sizes)

    # Get the distribution of the differences of the metric
    sample_metric = metric(series)
    diff_samples = [samp .- sample_metric for samp in subsamples]

    # Compute the differences between consecutive quantiles of the empirical estimatate of
    # the samples
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

    quant_diffs = [-1] # just so it lives outside of the while scope and triggers the loop.

    while !all([all(q .> 0) for q in quant_diffs])
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

    # compute the variances of the log of the response variables
    # avoid floating point errors by flooring to 0 if below machine precision.
    vars = map(y) do _y
        v = var(log.(_y))
        return v < eps() ? 0 : v
    end

    # weight each point inversely to its variance
    weights = 1 ./ vars

    if !all(isfinite.(weights))
        n = count(.!isfinite.(weights))
        @warn("$n non-finite weights arose in computing log-log slope.")
    end

    # Use matrix inversion to compute the slope, i.e. converenge rate
    beta = sum(weights .* diff_y .* diff_x) / sum(weights .* (diff_x .^ 2))
end

"""
    compute_distance(cdf1, cdf2, locations)

Computes the unormalised squared L2 distance between two functions at a discrete set of
`locations`.
"""
function compute_distance(cdf1, cdf2, locations)
    return sum(x -> x^2, (cdf1.(locations) .- cdf2.(locations)))
end

"""
    default_sizemin(f)

Default value of minimum block size for metric `f`. Uses the minimum possible value of 4 for
all metrics, except the ones that involve expected shortfall or expected windfall, as the
minimum for those (if computed at 5%) must be 40.
"""
default_sizemin(f) = 4
default_sizemin(f::typeof(ew_over_es)) = 40
default_sizemin(f::typeof(mean_over_es)) = 40
default_sizemin(f::typeof(median_over_es)) = 40
default_sizemin(f::typeof(expected_shortfall)) = 40
default_sizemin(f::typeof(expected_windfall)) = 40

"""
    adaptive_block_size(
        metric::Function, series;
        sizemin=default_sizemin(metric),
        sizemax=ceil(Int, 0.8 * length(series)),
        sizestep=2,
        circular=false,
        numpoints=50,
    )

Estimate the optimal block size for computing the subsampled confidence interval of `metric`
over `series` using the adaptive method proposed in "Friedrich Götze and Alfredas Račkauskas
Lecture Notes-Monograph Series Vol. 36, State of the Art in Probability and Statistics
(2001), pp. 286-309".

Optimal size is searched over the range `sizemin:sizestep:sizemax`. If any of those
quantities is an odd number, the next integer is taken.

If `circular`, blocks wrap around the end of the `series`.

`numpoints` controls the number of points over which the empirical CDFs are compared.
"""
function adaptive_block_size(
    metric::Function, series;
    sizemin=default_sizemin(metric),
    sizemax=ceil(Int, 0.8 * length(series)),
    sizestep=2,
    circular=false,
    numpoints=50,
)
    sizemin = isodd(sizemin) ? sizemin + 1 : sizemin
    sizemax = isodd(sizemax) ? sizemax + 1 : sizemax
    sizestep = isodd(sizestep) ? sizestep + 1 : sizestep

    block_sizes = collect(sizemin:sizestep:sizemax)
    half_block_sizes = Int.(block_sizes ./ 2)

    # Build blocks for each size
    block_sets = block_subsample.(Ref(series), block_sizes; circular=circular)
    half_block_sets = block_subsample.(Ref(series), half_block_sizes; circular=circular)

    # Build metrics series
    metric_series = map(x -> metric.(x), block_sets)
    half_metric_series = map(x -> metric.(x), half_block_sets)

    # Compute empirical CDFs
    ecdfs = ecdf.(metric_series)
    half_ecdfs = ecdf.(half_metric_series)

    # Define locations to compute distance between CDFs
    # Makes it such that all CDFs are evaluated at the same points
    # Focuses on the central region of the support, because that should be modelled better
    center = median(mean.(metric_series))
    width = median(std.(metric_series))
    if width == 0
        @warn """
            Failed to estimate block size. The metric value is identical for every block.
            Returning smallest possible block size.
        """
        @info "Standard deviation of metrics over blocks: $(std.(metric_series))."
        return sizemin
    end
    locations = (center - width):(2.0 * width) / numpoints:(center + width)

    Δs = compute_distance.(ecdfs, half_ecdfs, Ref(locations))

    return block_sizes[findmin(Δs)[2]]
end

"""
    adaptive_block_size(
        metric::Function, series1, series2;
        sizemin=default_sizemin(metric),
        sizemax=ceil(Int, 0.8 * length(series2)),
        sizestep=2,
        circular=false,
        numpoints=50,
    )

Estimate the optimal block size for computing the subsampled confidence interval of the
difference in `metric` over `series1` and `series2` using the adaptive method proposed in
"Friedrich Götze and Alfredas Račkauskas Lecture Notes-Monograph Series Vol. 36, State of
the Art in Probability and Statistics (2001), pp. 286-309".
"""
function adaptive_block_size(
    metric::Function, series1, series2;
    sizemin=default_sizemin(metric),
    sizemax=ceil(Int, 0.8 * length(series2)),
    sizestep=2,
    circular=false,
    numpoints=50,
)
    length(series1) != length(series2) && throw(DimensionMismatch(
        "Both series must have the same length. Got $(length(series1)) and $(length(series2))."
    ))
    # Pair observations
    paired_series = collect(zip(series1, series2))
    # Define metric from R² to R
    diff_metric = x -> metric(getfield.(x, 1)) - metric(getfield.(x, 2))
    return adaptive_block_size(
        diff_metric::Function, paired_series;
        sizemin=sizemin,
        sizemax=sizemax,
        sizestep=sizestep,
        circular=circular,
        numpoints=numpoints,
    )
end

"""
    estimate_block_size(
        metric::Function, series;
        α=0.05,
        β=nothing,
        sizemin=default_sizemin(metric),
        sizemax=ceil(Int, 0.8 * length(series)),
        sizestep=1,
        blocksvol=2,
    )

Estimate optimal block size for computing confidence intervals at a level `α` for `metric`
over a `series` by minimising their volatility. Optimal size is searched over the range
`sizemin:sizestep:sizemax` and volatility is computed using `blocksvol` values above and
below a given block size. Confidence intervals are computed assuming a convergence rate of
`b^β`. If `β=nothing`, the rate is estimated via [`estimate_convergence_rate`](@ref).

For details on this procedure, see chapter 9 of "Politis, Dimitris N., Joseph P. Romano, and
Michael Wolf. Subsampling. Springer Science & Business Media, 1999."

Returns the optimal block size, the (possibly estimated) `β` and the confidence interval
at level `α` with the optimal block size.

* NOTE: The estimation method implemented here is obsolete and shown to be less efficient
(see https://docs.google.com/document/d/1R6eIpZakzAJZHnO8JcxrCfcBbFdYjWfuEk29rS8nVYk/edit#heading=h.8495mh10rdjq).
Unless you know exactly what you are doing, [`adaptive_block_size`](@ref) should be used
instead.
"""
function estimate_block_size(
    metric::Function, series;
    α=0.05,
    β=nothing,
    sizemin=default_sizemin(metric),
    sizemax=ceil(Int, 0.8 * length(series)),
    sizestep=1,
    blocksvol=2,
)
    β = isnothing(β) ? estimate_convergence_rate(metric, series) : β
    block_sizes = collect(sizemin:sizestep:sizemax)

    # compute CIs for each block size
    cis = subsample_ci.(Ref(metric), Ref(series), block_sizes; α=α, β=β)

    # obtain lower and upper bounds
    lows = first.(cis)
    ups = last.(cis)

    # determine block ranges to be used for computing the std of the CI bounds
    block_ranges = Metrics.block_subsample(1:length(cis), 2 * blocksvol + 1)

    # compute volatility indexes over the block_ranges
    vols = [std(lows[bs]) + std(ups[bs]) for bs in block_ranges]

    # Find the index of minimum volatility
    vmin, imin = findmin(vols)

    # Indent imin by blocksvol to give back the index in the cis array
    return block_sizes[imin + blocksvol]
end

"""
    default_β(f)

Default value of exponent of the convergence rate for metric `f`. If unknown, returns
`nothing`.
"""
default_β(f) = nothing
# Metrics for which we have computed the β values.
# See: https://docs.google.com/document/d/12g0uPpcWva-HpXi6OoPnR2FysqMCjJoOqJcpqkLzt3E
default_β(f::typeof(mean)) = 0.5
default_β(f::typeof(median)) = 0.5
default_β(f::typeof(mean_over_es)) = 0.5
default_β(f::typeof(median_over_es)) = 0.5
default_β(f::typeof(expected_shortfall)) = 0.5
default_β(f::typeof(expected_windfall)) = 0.5
default_β(f::typeof(ew_over_es)) = 0.5


"""
    subsample_ci(
        metric::Function, series;
        α=0.05,
        β=default_β(metric),
        sizemin=default_sizemin(metric),
        sizemax=ceil(Int, 0.8 * length(series)),
        sizestep=1,
        numpoints=50,
        studentise=false,
        circular=false,
        kwargs...
    )

Compute confidence interval for `metric` over a `series` at a level `α` and convergence rate
`b^β` by estimating the block size via [`adaptive_block_size`](@ref).

The `sizemin`, `sizemax`, `sizestep`, and `numpoints` keyword arguments are passed to
[`adaptive_block_size`](@ref).
The remaining `kwargs` are passed to [`estimate_convergence_rate`](@ref) (if `β` is
`nothing`) and [`subsample_ci`](@ref).
If `β=nothing`, the rate is estimated via [`estimate_convergence_rate`](@ref).

If `studentise`, the roots are studentised using the unbiased sample standard deviation. See
chapters 2 and 11 of "Politis, Dimitris N., Joseph P. Romano, and
Michael Wolf. Subsampling. Springer Science & Business Media, 1999." for the theory. For
heavy tailed data, it is recommended to use this option.

If `circular`, the subsampled blocks wrap around the end of `series`.

!!! note
    Since `α` is the _level_ of the test, we expect the true value to lie in `1-α` of the CIs.

!!! warning "Default β"
    If the `β` keyword is not provided it defaults to `default_β(metric)`.
    For anonymous function [`default_β`](@ref) will always be `nothing`.
    It is important to be aware of this when passing an anonymous function,
    for example when using `do`-block syntax to define the metric.
"""
function subsample_ci(
    metric::Function, series;
    α=0.05,
    β=default_β(metric),
    sizemin=default_sizemin(metric),
    sizemax=ceil(Int, 0.8 * length(series)),
    sizestep=1,
    numpoints=50,
    studentise=false,
    circular=false,
    kwargs...
)
    β = isnothing(β) ? estimate_convergence_rate(metric, series; kwargs...) : β
    block_size = adaptive_block_size(
        metric,
        series;
        sizemin=sizemin,
        sizemax=sizemax,
        sizestep=sizestep,
        numpoints=numpoints
    )

    return subsample_ci(
        metric, series, block_size;
        α=α, β=β, studentise=studentise, circular=circular, kwargs...
    )
end

"""
    subsample_ci(
        metric::Function, series, block_size;
        α=0.05, β=default_β(metric), studentise=false, circular=false, kwargs...
    )

Compute confidence interval for `metric` over a `series` at a level `α` using a `block_size`
and convergence rate `b^β`. If `β=nothing`, the rate is estimated via
[`estimate_convergence_rate`](@ref) which accepts the `kwargs`.

If `studentise`, the roots are studentised using the unbiased sample standard deviation. See
chapters 2 and 11 of "Politis, Dimitris N., Joseph P. Romano, and
Michael Wolf. Subsampling. Springer Science & Business Media, 1999." for the theory. For
heavy tailed data, it is recommended to use this option.

If `circular`, the subsampled blocks wrap around the end of `series`.

Returns the confidence interval as `Closed` `Interval`.
"""
function subsample_ci(
    metric::Function, series, block_size;
    α=0.05, β=default_β(metric), studentise=false, circular=false, kwargs...
)
    # apply metric to subsampled series
    blocks = block_subsample(series, block_size; circular=circular)
    metric_series = metric.(blocks)

    # Internal function to compute a measure of spread for studentisation
    function spread(series)
        if series isa Vector{<:Tuple} # Will happen when we are subsampling differences
            return sqrt(var(getindex.(series, 1)) + var(getindex.(series, 2)))
        else
            return std(series)
        end
    end

    # By default, Statistics uses the unbiased version of `var`.
    σ_b = studentise ? spread.(blocks) : ones(length(blocks))
    # estimate convergence rates
    β = isnothing(β) ? estimate_convergence_rate(metric, series; kwargs...) : β
    n = length(series)
    τ_b = block_size ^ β
    τ_n = n ^ β
    σ_n = studentise ? spread(series) : 1.0
    # compute sample metric
    sample_metric = metric(series)
    # center and scale metrics
    metric_series = (metric_series .- sample_metric) * τ_b ./ σ_b
    # compute lower and upper bounds
    lower = quantile(metric_series, α / 2)
    upper = quantile(metric_series, 1 - (α / 2))
    # apply location and scale estimates
    lower_corrected = sample_metric - upper / τ_n * σ_n
    upper_corrected = sample_metric - lower / τ_n * σ_n
    return Interval(lower_corrected, upper_corrected)
end

