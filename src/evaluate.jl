# Trait for if metric applies to a whole dataset or just one single thing
abstract type ObsMetric end
struct SingleObsMetric <:ObsMetric end
struct MultipleObsMetric <:ObsMetric end

obs_quantity(x::T) where T = obs_quantity(T)

function split_into_obs(metric, data; obsdim=nothing)
    split_into_obs(obs_quantity(metric), data; obsdim=obsdim)
end

split_into_obs(::SingleObsMetric, data; obsdim=nothing) = data
function split_into_obs(::MultipleObsMetric, data; obsdim=nothing)
    if obsdim == nothing
        obsdim = _default_obsdim(data)
    end
    # use `obsview` rather than `eachobs` as we want to be able to
    # e.g. `collect` the result potentially. So can't reuse the buffer.
    return obsview(data, obsdim)
end

_default_obsdim(x) = MLDataUtils.default_obsdim(x)
_default_obsdim(::AbstactArray) = 1  # Talk to Eric P. if you disagree
function _default_obsdim(x::NamedDimsArray)
    dimnames = NamedDims.names(x)
    dim = if :obs ∈ dimnames
        :obs
    elseif :observations ∈ dimnames
        :observations
    elseif :samples ∈ dimnames
        :samples
    end
    return NamedDims.numerical_dim(dim)
end

"""
    evaluate(metric, grounds_truths, predictions)

Compute a metric with truths `grounds_truths` and predictions `predictions`

### Keyword Arguments
 - `obsdim`: Which dimension of a array contains the observation.
Determined automatically if not provided. Ignored if a metric is not defined
across multiple observations.
"""
function evaluate(metric, grounds_truths, predictions; obsdim=nothing)
    return metric(
        split_into_obs(metric, grounds_truths; obsdim=obsdim)
        split_into_obs(metric, predictions; obsdim=obsdim)
    )
end
