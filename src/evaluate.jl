"""
    evaluate(metric, args...; [obsdim], kwargs...)

Compute a `metric` with the given arguments using the metric's `ObsArrangement`.
This rearranges the `args` (if required) before passing to the `metric`.
One should look at the documentation for the individual `metric`.

Keyword Arguments:
 - `obsdim`: Which dimension of an array contains the observation. Determined automatically
    if not provided. Ignored if a metric is not defined across multiple observations.
 - all other `kwargs` are passed to the `metric` function.
"""
function evaluate(metric, args...; obsdim=nothing, kwargs...)
    return metric(
        (organise_obs(metric, arg; obsdim=obsdim) for arg in args)...;
        kwargs...
    )
end
