macro _dimcheck(condition::Expr)
    explanation = ""
    if condition.head == :call && length(condition.args) == 3
        comparison, lhs, rhs = condition.args
        explanation = :(string(
            " It is not the case that ",
            $lhs, " ", $comparison, " ", $rhs, "."
        ))
    elseif condition.head == :comparison
        explanation = :(string(
            " It is not the case that ",
            $(condition.args...), "."
        ))
    end

    quote
        if !$(esc(condition))
            throw(DimensionMismatch(string(
                "Dimensions of the parameters don't match: ",
                $(string(condition)),
                ".",
                $(esc(explanation))
            )))
        end
    end
end

# If I understand this page correctly, `eachrow` should exists for `1.1`
# https://github.com/JuliaLang/julia/commit/6b0429181142eabb441c1febf0ae286f559b2f32
if VERSION < v"1.1"
    eachrow(A::AbstractVecOrMat) = (view(A, i, :) for i in axes(A, 1))
end


"""
    _match(a::AxisArray, b::AxisArray) -> Tuple{a::AxisArray, b::AxisArray}
    _match(a::AxisArray, d::IndexedDistribution) -> Tuple{a::AxisArray, d::IndexedDistribution}

Match the axis values and/or indices of the provided arguments such that their values are in
the same order. This can be applied to pairs of `AxisArray`s or an `AxisArray` with an
`IndexedDistribution`.

Arguments:
  - `a::AxisArray`: An array of data whose axes are indexed by some values
  - `b::AxisArray`: An array of data whose axes are indexed by some values
  - `d::IndexedDistribution`: A distribution indexed by some values, usually produced as the
  output of a forecaster

Returns:
  - `Tuple{a::AxisArray, b::AxisArray}`: A tuple of the same input data but with axes aligned
  and ordered.
  - `Tuple{a::AxisArray, d::IndexedDistribution}`: A tuple of the same input data with their
  axes and indices aligned. The order is the same as the index in the input IndexedDistribution

Throws:
  - `ArgumentError`: If the AxisArrays do not have the same orientation
  - `ArgumentError`: If the AxisArrays do not have the same axis values
  - `ArgumentError`: If the axis values of the AxisArray do not match the indices of the
  IndexedDistribution

```jldoctest; setup = :(using AxisArrays, Distributions, IndexedDistributions, Metrics)
julia> a = AxisArray([1, 2, 3], Axis{:node}(["b", "a", "c"]));

julia> b = AxisArray([1, 2, 3], Axis{:node}(["c", "b", "a"]));

julia> Metrics._match(a, b)
([2, 1, 3], [3, 2, 1])
```
"""
function _match(a::AxisArray, d::IndexedDistribution)
    # marginal_gaussian_loglikelihood, etc. can accept multiple observations at a time
    # The stricter condition: size(a) == size(d) should be picked up by any metric that
    # requires it
    @_dimcheck first(size(a)) == first(size(d))

    dist, names = parent(d), index(d)
    index_dim = findfirst(Ref(sort(names)) .== sort.(axisvalues(a)))

    if index_dim isa Nothing
        throw(ArgumentError(
            "Index and axis values do not match: index = $(index(d)), axis = $(axisvalues(a))"
        ))
    end

    if axisvalues(a)[index_dim] == index(d)
        # if the orders match already, return the inputs as they were
        return a, d
    else
        # if the orders don't match, align the `AxisArray` to match the order in the
        # `IndexedDistribution` and don't change the `IndexedDistribution`. (This is because
        # altering the  index order of a distribution is hard to maintain the exact type of
        # of the underlying distribution and the specific PDMats)

        # re-organise the AxisArray
        new_a_data = copy(a.data)
        old_axes = AxisArrays.axes(a)
        new_axes = (
            old_axes[1:index_dim-1]...,
            Axis{axisnames(a)[index_dim]}(names),
            old_axes[index_dim+1:end]...,
        )
        old_idxs, new_idxs = AxisArrays.indexmappings(old_axes, new_axes)
        new_a_data[new_idxs...] = a.data[old_idxs...]
        matched_a = AxisArray(new_a_data, new_axes)

        return matched_a, d
    end
end

function _match(a::AxisArray, b::AxisArray)
    @_dimcheck size(a) == size(b)

    # if AxisArrays already match then short-circtuit the rest
    if (axisnames(a) == axisnames(b)) && (axisvalues(a) == axisvalues(b))
        return a, b
    # check that axis orientation is the same
    elseif axisnames(a) != axisnames(b)
        throw(ArgumentError(
            "AxisArray orientations do not match: "*
            "axisnames(a) = $(axisnames(a)), axisnames(b) = $(axisvalues(b))"
        ))
    # check that axis values are the same
    elseif sort.(axisvalues(a)) != sort.(axisvalues(b))
        throw(ArgumentError(
            "AxisArray axis values do not match: "*
            "axisnames(a) = $(sort.(axisvalues(a))), axisnames(b) = $(sort.(axisvalues(b)))"
        ))
    else

        pa = sortperm.(axisvalues(a))
        pb = sortperm.(axisvalues(b))

        return a[pa...], b[pb...]
    end
end

_match(a, d) =  a, d

# NOTE: the TDist provided by Distributions.jl (https://github.com/JuliaStats/Distributions.jl/blob/997f2bbdc7d40982ec0a90e9aba7d0124b78bb52/src/univariate/continuous/tdist.jl)
#   doesn't have non-standard location and scale parameter, and hence define our own type.
struct GenericTDist{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    μ::T
    σ::T
    GenericTDist{T}(ν::T, µ::T, σ::T) where {T<:Real} = new{T}(ν, µ, σ)
end

"""
extract the covariance matrix in its original `AbstractPDMat` type
"""
# TODO: should I force the output type to be `C`? If so, what whould be the output if `df < 2`?
using PDMatsExtras.PDMats # I know I know - it won't be in the final version
function _cov(dist::MvNormal{M,C}) where {M, C<:AbstractPDMat}
    return dist.Σ
end

function _cov(dist::GenericMvTDist)
    return dist.df>2 ? (dist.df/(dist.df-2))*d.Σ : NaN*ones(d.dim, d.dim)
end

_cov(dist::IndexedDistribution) = _cov(parent(dist))

"""
extract the scale matrix in its original `AbstractPDMat` type
"""
function _scale(dist::Union{MvNormal, GenericMvTDist})
    return dist.Σ
end

_scale(dist::Union{Normal, GenericTDist} ) = dist.σ

_scale(dist::IndexedDistribution) = _scale(parent(dist))

# the following should probably go to `PDMatsExtras.jl`
# NOTE: the parameterisation to scale up the Woodbury matrix is not unique
#   `*` for PDMat, PSDMat and PDiagMat were already impplemented
import Base: *
*(a::WoodburyPDMat, c::T) where T<:Real = WoodburyPDMat(
    a.A,
    a.D * c,
    a.S * c
)
