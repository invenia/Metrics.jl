"""
    potential_payoff(y_true, y_pred)

Approximate the potential payoff of the predicted deltas `y_pred`, in units of USD/MWh, from
a simplified Portfolio Optimization (PO) given the true deltas `y_true`.

Consider the objective function in Markowitz PO: ``w^* = max_{w} μw - γwΣw'``.
In the absence of all constraints, maximising this function yields an optimal volume
`w^*` that is proportional to `Σ^{-1}μ`.

Hence, given the true deltas ``Δ``, the potential payoff is also proportional to `Σ^{-1}μ`:

```math
p = dot(Σ^{-1}μ, Δ) / norm(Σ^{-1}μ, 1)
```

where the denominator ``norm(Σ^{-1}μ, 1)`` is introduced to mimick a constraint on the total
volume. Note, however, that this formulation ignores other constraints that might be present
in our actual PO, such as net total volume, volume-per-node, price impact, fees, etc.

Methods for this metric are defined only for a single forecast and, depending on the
variables provided, make strong assumptions about what kind of PO is performed.

* Univariate / Number - PO performed on one node for one hour with fixed total MW
* Multivariate / Vector - PO performed on all nodes for one hour with fixed hourly total MW
* Matrixvariate / Matrix - PO performed on all nodes for all hours with fixed daily total MW

Only the Multivariate PO is currently performed in EIS. Matrixvariate PO is something we
may wish to explore in future. Univariate PO is included for completeness and simply returns
``sgn(μ)Δ``.

"""
function potential_payoff(y_true, y_pred)
    @_dimcheck size(y_true) == size(y_pred)
    p = dot(y_pred, y_true) / norm(y_pred, 1)

    # ensure potential_payoff is stable when predicting 0s
    return isnan(p) ? 0 : p

end

"""
    potential_payoff(y_true, y_pred::Sampleable{Univariate})

Compute the [`potential_payoff`](@ref) for a single node on a single hour with fixed total MW.
This method returns `sgn(y_pred) * y_true` and so rewards predicting the correct sign `y_true`.
"""
function potential_payoff(y_true, y_pred::Sampleable{Univariate})
    @_dimcheck size(y_true) == size(y_pred)
    _y_pred = mean(y_pred) / var(y_pred)
    return potential_payoff(y_true, _y_pred)
end

"""
    potential_payoff(y_true, y_pred::Sampleable{Multivariate})

Compute the [`potential_payoff`](@ref) for all nodes on a single hour with fixed hourly
total MW. This method most faithfully resembles our current production PO framework but
without additional constraints such as net total volume, volume-per-node, price impact, etc.
"""
function potential_payoff(y_true, y_pred::Sampleable{Multivariate})
    @_dimcheck size(y_true) == size(y_pred)
    _y_pred, _y_true = _match(y_pred, y_true)

    # TODO: Ideally this line would require that \ is defined on AxisArrays
    # since it currently isn't we have to call parent() no the object
    _y_pred = parent(cov(_y_pred)) \ parent(mean(_y_pred))
    return potential_payoff(_y_true, _y_pred)
end

"""
    potential_payoff(y_true, y_pred::Sampleable{Matrixvariate})

Compute the [`potential_payoff`](@ref) for all nodes and all hours with fixed daily total MW.
This method is not implemented in production but is one we may explore in future.

The Matrixvariate distribution is first vectorised and [`potential_payoff`](@ref) is then
called on the Multivariate distribution instance.
"""
function potential_payoff(y_true, y_pred::Sampleable{Matrixvariate})
    return potential_payoff(vec(y_true), vec(y_pred))
end

ObservationDims.obs_arrangement(::typeof(potential_payoff)) = SingleObs()
