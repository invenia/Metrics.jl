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
    elseif obsdim isa Integer
        obsdim = MLDataUtils.ObsDim.Constant{obsdim}()
    end
    # use `obsview` rather than `eachobs` as we want to be able to
    # e.g. `collect` the result potentially. So can't reuse the buffer.
    return obsview(data, obsdim)
end

_default_obsdim(x) = MLDataUtils.default_obsdim(x)
_default_obsdim(::AbstractArray) = MLDataUtils.ObsDim.First()  # Talk to Eric P. if you disagree
function _default_obsdim(x::NamedDimsArray{L}) where L
    obsnames = (:obs, :observations, :samples)
    used = findfirst(in(L), obsnames)
    if used===nothing
        throw(DimensionMismatch(string(
            "No observation dimension found. Provided dimension names = $L, ",
            "valid observation dimension names = $obsnames"
        )))
    end
    dim = @inbounds obsnames[used]
    numerical_dim = NamedDims.dim(L, dim)
    return MLDataUtils.ObsDim.Constant{numerical_dim}()
end

"""
    evaluate(metric, grounds_truths, predictions, args...)

Compute a metric with truths `grounds_truths` and predictions `predictions`

### Keyword Arguments
 - `obsdim`: Which dimension of a array contains the observation.
Determined automatically if not provided. Ignored if a metric is not defined
across multiple observations.
"""
function evaluate(metric, grounds_truths, predictions, args...; obsdim=nothing)
    return metric(
        split_into_obs(metric, grounds_truths; obsdim=obsdim),
        split_into_obs(metric, predictions; obsdim=obsdim),
        args...
    )
end
