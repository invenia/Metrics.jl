"""
    expected_shortfall(returns, α::Float64)

Calculate the expected shortfall `-𝔼[ r_p | r_p ≤ q_risk_level(r_p) ]`, where `r_p` is
the portfolio return and `q_risk_level(r_p)` is the `α`-quantile of the
distribution of `r_p`.

NOTE: Expected shortfall is the _negative_ of the average of the bottom
`α`-quantile of returns. Assuming average is positive for all `α`, then
it is good to _minimise_ expected shortfall.
"""
function expected_shortfall(returns, α::Float64)

    0 < α < 1 || throw(ArgumentError("α=$α is not between 0 and 1."))

    last_index = floor(Int, α * length(returns))
    last_index > 0 || throw(
        ArgumentError(string(
                "length(returns)=$(length(returns)) not enough elemnts for α=$α.",
                " Min length(returns)=$(ceil(Int, 1/α))"
        ))
    )

    return mean(partialsort(returns, 1:last_index))
end

obs_arrangement(::typeof(expected_shortfall)) = IteratorOfObs()
