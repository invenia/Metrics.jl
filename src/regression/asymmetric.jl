"""
    asymmetric_absolute_error(y_true, y_pred; underpredict_penalty=0.0)

Calculate an asymmetric absolute error between `y_true` and `y_pred`.
If `underpredict_penalty > 0` then underpredictions (`y_pred < y_true`) are penalised more
than overpredictions.

The error is given by:
```
is_over = (y_pred > y_true)
error = |0.5 + 0.5 * underpredict_penalty - is_over| * |y_pred - y_true|
```

Special cases:
- `underpredict_penalty = 0` (default): under and over predictions will be penalised
    symmetrically.
- `underpredict_penalty = 1`: overpredictions will give an error of zero.
- `underpredict_penalty = -1`: underpredictions will give an error of zero.

# Arguments:
- `y_true, y_pred`: true and predicted values.

# Keyword arguments:
- `underpredict_penalty`: Should be between -1 and 1. If >0, penalise underpredictions
    more than overpredictions; if <0 then the reverse.
"""
function asymmetric_absolute_error(y_true::Number, y_pred::Number; underpredict_penalty=0.0)
    if !(-1 ≤ underpredict_penalty ≤ 1)
        throw(DomainError("underpredict_penalty must be between -1 and 1"))
    end

    # α → 0 means we penalise overpredictions more
    # α → 1 means we penalise underpredictions more
    # α = 0.5 means no extra penalty either way
    α = 0.5 + 0.5 * underpredict_penalty

    # underpredict_penalty penalises the case where y_pred < y_true
    # i.e. e := y_pred - y_true < 0
    e = y_pred - y_true

    is_over = (e > 0)

    return abs(α - is_over) * abs(e)
end

ObservationDims.obs_arrangement(::typeof(asymmetric_absolute_error)) = SingleObs()
