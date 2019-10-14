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
    _match(a::AxisArray, d::IndexedDistribution) -> a, d

Sort the indices of the provided `AxisArray` and `IndexedDistribution` such that they align.

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
function _match(a::AxisArray, d::IndexedDistribution)

    dist, names = parent(d), index(d)
    index_dim = findfirst(Ref(sort(names)) .== sort.(axisvalues(a)))

    if isnothing(index_dim)
     throw(ArgumentError(
        "Index and axis values do not coincide: index = $(index(d)), axis = $(axisvalues(a))"
    ))
    end

    # determine the correct ordering
    pd = sortperm(names)
    pa = sortperm(axisvalues(a)[index_dim])

    μ, Σ = mean(dist), cov(dist)  # params() returns a PDMat which can't be re-sorted
    sorted_d = IndexedDistribution(MvNormal(vec(μ[pd, :]), Σ[pd, pd]), names[pd])

    return a[pa], sorted_d
end

_match(a, d) =  a, d
