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
    _match(a::KeyedArray, b::KeyedArray) -> Tuple{a::KeyedArray, b::KeyedArray}
    _match(a::KeyedArray, d::KeyedDistribution) -> Tuple{a::KeyedArray, d::KeyedDistribution}

Match the axis keys and/or indices of the provided arguments such that their values are in
the same order. This can be applied to pairs of `KeyedArray`s or an `KeyedArray` with an
`KeyedDistribution`.

Returns:
  - `Tuple{a::KeyedArray, b::KeyedArray}`: A tuple of the same input data but with dims aligned
  and ordered.
  - `Tuple{a::KeyedArray, d::KeyedDistribution}`: A tuple of the same input data with their
  dims and keys aligned. The order is the same as the keys in the input `KeyedDistribution`.

Throws:
  - `ArgumentError`: If the KeyedArrays do not have the same orientation.
  - `ArgumentError`: If the KeyedArrays do not have the same axiskeys.
  - `ArgumentError`: If the axiskeys of the KeyedArray do not match the indices of the
  KeyedDistribution.

```jldoctest; setup = :(using AxisKeys, Distributions, KeyedDistributions, Metrics)
julia> a = KeyedArray([1, 2, 3]; node=["b", "a", "c"]);

julia> b = KeyedArray([1, 2, 3]; node=["c", "b", "a"]);

julia> Metrics._match(a, b)
([2, 1, 3], [3, 2, 1])
```
"""
function _match(a::KeyedArray, d::KeyedDistribution)
    # marginal_gaussian_loglikelihood, etc. can accept multiple observations at a time
    # The stricter condition: size(a) == size(d) should be picked up by any metric that
    # requires it
    @_dimcheck first(size(a)) == first(size(d))

    names = only(axiskeys(d))
    dim = findfirst(Ref(sort(names)) .== sort.(axiskeys(a)))

    if dim isa Nothing
        throw(ArgumentError(
            "axiskeys do not match: distribution: $(axiskeys(d)), array: $(axiskeys(a))"
        ))
    end

    if axiskeys(a)[dim] == names
        # if the orders match already, return the inputs as they were
        return a, d
    else
        return a(names), d
    end
end

function _match(a::KeyedArray, b::KeyedArray)
    @_dimcheck size(a) == size(b)

    # if KeyedArrays already match then short-circtuit the rest
    if (dimnames(a) == dimnames(b)) && (axiskeys(a) == axiskeys(b))
        return a, b
    # check that axis orientation is the same
    elseif dimnames(a) != dimnames(b)
        throw(ArgumentError(
            "KeyedArray orientations do not match: "*
            "dimnames(a) = $(dimnames(a)), dimnames(b) = $(dimnames(b))"
        ))
    # check that axis values are the same
    elseif sort.(axiskeys(a)) != sort.(axiskeys(b))
        throw(ArgumentError(
            "KeyedArray axis values do not match: "*
            "axiskeys(a) = $(sort.(axiskeys(a))), axiskeys(b) = $(sort.(axiskeys(b)))"
        ))
    else

        pa = sortperm.(axiskeys(a))
        pb = sortperm.(axiskeys(b))

        return a[pa...], b[pb...]
    end
end

_match(a, d) =  a, d
