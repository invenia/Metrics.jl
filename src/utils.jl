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
