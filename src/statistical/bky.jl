"""
    bky_test(pvalues::Vector, q=0.05)

Perform the Bejamini-Krieger-Yekutieli procedure for multiple-hypothesis testing over a
vector of `pvalues` at confidence level `q`.

Described in Definition 7 of "Adaptive linear step-up procedures that control the false
discovery rate", Benjamini Y., Krieger A.M., Yekutieli D., 2006.

Returns a vector of `Bool`s - each corresponding to one p-value - for which `true` means
the null hypothesis is rejected and `false` means the null hypothesis is not rejected.
"""
function bky_test(pvalues::Vector, q=0.05)

    0 < q < 1 || throw(DomainError(q, "Confidence level q must be in the interval [0, 1])"))

    n = length(pvalues)

    # Sort values and save original order
    sorted_pvalues = sort(pvalues)
    original_order = sortperm(pvalues)

    # Initiate loop parameters
    current_p_idx = 0
    num_significant_ps = 0
    find_cutoff = true

    while find_cutoff && current_p_idx <= n

        current_p_idx += 1
        remaining_pvals = sorted_pvalues[current_p_idx:end]

        # Compute the required thresholds for the remaining pvalues to be significant
        thresholds = [l * q / (n + 1 - current_p_idx * (1 - q)) for l in current_p_idx:n]

        # Check if any pvalues are still significant at this threshold level
        # If none are then we can safely initiate an early stopping condition
        if any(remaining_pvals .<= thresholds)
            num_significant_ps += 1
        else
            find_cutoff = false
        end

    end

    # By default we don't reject the null hypothesis for all pvalues
    reject_null = zeros(Bool, n)

    # Where we have found significant pvalues we can reject the null hypothesis
    # These are by definition the smallest num_significant_ps as originally ordered
    if num_significant_ps > 0
        rejected = original_order[1:num_significant_ps]
        reject_null[rejected] .= true
    end

    return reject_null
end

obs_arrangement(::typeof(bky_test)) = MatrixColsOfObs()
