using Base: @deprecate

@deprecate(
    estimate_convergence_rate(series, metric::Function, args...; kwargs...),
    estimate_convergence_rate(metric, series, args...; kwargs...),
)

@deprecate(
    estimate_block_size(series, metric::Function; kwargs...),
    estimate_block_size(metric, series; kwargs...),
)

@deprecate(
    subsample_ci(series, metric::Function, args...; kwargs...),
    subsample_ci(metric, series, args...; kwargs...)
)
