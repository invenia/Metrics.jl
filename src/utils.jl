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
  - `Tuple{a::AxisArray, b::AxisArray}`: A tuple of the same input data but with axes aligned.
  - `Tuple{a::AxisArray, d::IndexedDistribution}`: A tuple of the same input data with their
  axes and indices aligned.

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

```jldoctest; setup = :(using AxisArrays, Distributions, IndexedDistributions, Metrics)
julia> a = AxisArray([1, 2, 3], Axis{:node}(["b", "a", "c"]));

julia> d = IndexedDistribution(MvNormal(ones(3)), ["c", "b", "a"]);

julia> Metrics._match(a, d)
([2, 1, 3], IndexedDistribution{Multivariate,Continuous,MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}}(FullNormal(
dim: 3
μ: [0.0, 0.0, 0.0]
Σ: [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
)
, ["a", "b", "c"]))
```
"""
function _match(a::AxisArray, d::IndexedDistribution{F, S, <:AbstractMvNormal}) where {F, S}
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
        return a, d
    else
        # re-organise the AxisArray
        pa = sortperm.(axisvalues(a))
        sorted_a = a[pa...]

        # re-organise the IndexedDistribution
        pd = sortperm(names)
        μ, Σ = mean(dist), cov(dist)  # params() returns a PDMat which can't be re-sorted
        sorted_d = IndexedDistribution(MvNormal(vec(μ[pd, :]), Σ[pd, pd]), names[pd])

        @assert axisvalues(sorted_a)[index_dim] == index(sorted_d)

        return sorted_a, sorted_d
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
