# Trait for if metric applies to a whole dataset or just one single thing
abstract type ObsArrangement end
# trait function is `obs_arrangement`

# These traits say how the given function wants to get its observations.
struct SingleObs <: ObsArrangement end
struct IteratorOfObs <: ObsArrangement end
struct ArraySlicesOfObs{D} <: ObsArrangement end
const MatrixRowsOfObs = ArraySlicesOfObs{1}
const MatrixColsOfObs = ArraySlicesOfObs{2}
# TODO: Consider adding `VectorOfObs`, which is like `collect` + `IteratorOfObs`
# this would let things that are type-constrained to `AbstractVector` work.
# At the cost of matrializing all the generators that slice up matrixes

## pre-trait
# trait re-dispatch
function arrange_obs(metric, data; obsdim=nothing)
    return arrange_obs(obs_arrangement(metric), data; obsdim=obsdim)
end

# never slice: (avoid even thinking about traits)
for T in (Distribution, Number, Symbol,)
    @eval arrange_obs(metric, data::$T; obsdim=nothing) = data
end

## Two arg forms: obsdim is optional, we may or may not need it.
arrange_obs(::SingleObs, data; obsdim=nothing) = data

for T in (Any, AbstractVector)
    @eval arrange_obs(::IteratorOfObs, obs_iter::$T; obsdim=nothing) = obs_iter

    @eval function arrange_obs(
        ::ArraySlicesOfObs{D},
        obs_iter::$T;
        obsdim=nothing
    ) where D
        # we assume all obs have same number of dimensions else nothing makes sense
        ndims_per_obs = ndims(first(obs_iter))

        if ndims_per_obs == 0  # it is a collection of scalars!
            # TODO: idk if this is best behavour, but it is useful for our likelyhoods
            # for the univariate case
            return collect(obs_iter)
        end

        # This should just be a mapreduce but that is slow
        # see https://github.com/JuliaLang/julia/issues/31137
        shaped_obs = Base.Generator(obs_iter) do obs
            new_shape = ntuple(ndims_per_obs + 1) do ii
                if ii < D
                    size(obs, ii)
                elseif ii > D
                    size(obs, ii-1)
                else  # ii = D
                    1
                end
            end
            # add singleton dim that we wil concatenate on
            Base.ReshapedArray(obs, new_shape, ())
        end
        return cat(shaped_obs...; dims=D)
    end
end

# Avoid ambiguity
for A in (IteratorOfObs, ArraySlicesOfObs)
    @eval function arrange_obs(arrangement::$A, data::AbstractArray; obsdim=nothing)
        if obsdim == nothing
            obsdim = _default_obsdim(data)
        end
        if data isa NamedDimsArray && obsdim isa Symbol
            obsdim = NamedDims.dim(data, obsdim)
        end

        return arrange_obs(arrangement, data, obsdim)
    end
end

## 3 arg forms: we know the obsdim we need and our current form may not agree
function arrange_obs(::IteratorOfObs, data::AbstractArray, obsdim::Integer)
    # This is basically eachslice from julia 1.1+
    return (selectdim(data, obsdim, ii) for ii in axes(data, obsdim))
end

function arrange_obs(
    ::ArraySlicesOfObs{D},
    data::AbstractArray{<:Any, N},
    obsdim::Integer
) where {D, N}

    if obsdim == D
        return data
    else
        perm = ntuple(N) do ii
            if ii == D
                return obsdim
            elseif ii == obsdim
                return D
            else
                return ii
            end
        end
        return PermutedDimsArray(data, perm)
    end
end


_default_obsdim(x) = 1  # for iterators
_default_obsdim(::AbstractArray) = 1  # Talk to Eric P. if you disagree
function _default_obsdim(x::NamedDimsArray{L}) where L
    obsnames = (:obs, :observations, :samples)
    used = findfirst(in(L), obsnames)
    if used === nothing
        throw(DimensionMismatch(string(
            "No observation dimension found. Provided dimension names = $L, ",
            "valid observation dimension names = $obsnames"
        )))
    end
    return @inbounds obsnames[used]
end

"""
    evaluate(metric, grounds_truths, predictions, args...)

Compute a metric with truths `grounds_truths` and predictions `predictions`

### Keyword Arguments
 - `obsdim`: Which dimension of a array contains the observation.
Determined automatically if not provided. Ignored if a metric is not defined
across multiple observations.
"""
function evaluate(metric, args...; obsdim=nothing, kwargs...)
    return metric(
        (arrange_obs(metric, arg; obsdim=obsdim) for arg in args)...;
        kwargs...
    )
end
