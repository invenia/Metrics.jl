"""
    evaluate(metric, args...; [obsdim], kwargs...)

Compute a metric with with the given arguements.
the `args` are passed to the `metric` function after rearragement (if required)
to match the expected observation structure used by the given `metric`.

One should thus look at the documentation for the individual metric

### Keyword Arguments
 - `obsdim`: Which dimension of a array contains the observation.
Determined automatically if not provided. Ignored if a metric is not defined
across multiple observations.
 - all other `kwargs` are passed to the `metric` function
"""
function evaluate(metric, args...; obsdim=nothing, kwargs...)
    return metric(
        (arrange_obs(metric, arg; obsdim=obsdim) for arg in args)...;
        kwargs...
    )
end

# These traits say how the given function wants to get its observations.
# trait function is `obs_arrangement`
abstract type ObsArrangement end

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


# These are basically scalar quantities and don't (generally) represent observations
# at all. e.g. threshold parameters.
# never slice: (avoid even thinking about traits)
for T in (Distribution, Number, Symbol,)
    @eval arrange_obs(metric, data::$T; obsdim=nothing) = data
end

## Two arg forms: obsdim is optional, we may or may not need it.

for T in (Any, AbstractVector)
    # Need a iterator and it aleady is an iterator so no need ot change
    @eval arrange_obs(::IteratorOfObs, obs_iter::$T; obsdim=nothing) = obs_iter

    # It is an iterator of observations and we need to arrange it into an Array
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

# If it is a single observation than never any need to rearrage
arrange_obs(::SingleObs, data; obsdim=nothing) = data

for A in (IteratorOfObs, ArraySlicesOfObs)
    # Handle non-1D AbstractArray data, this means we need to know the obsdim
    # This method fills in the obsdim, if required, to the default
    # then redispatches to the 3 arg form below.
    # Filling in the observation dimension is the same regardless of if targetting
    # IteratorOfObs, ArraySlicesOfObs
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
## These are only needed for (non 1D) arrays

# Slice up the array to get an iterator of observations
function arrange_obs(::IteratorOfObs, data::AbstractArray, obsdim::Integer)
    # This is basically eachslice from julia 1.1+
    return (selectdim(data, obsdim, ii) for ii in axes(data, obsdim))
end

# Permute the array so the observations are on the right dimension
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
