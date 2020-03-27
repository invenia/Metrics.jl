"""
    dm_mean_test(diffs; bandwidth=21)

Compute the Diebold-Mariano p-value for the mean of a series of loss differentials, `diffs`.
In practice, we take `diffs` as the difference between two financial return timeseries.
The null hypothesis is that the mean difference is zero.

The `bandwidth` keyword is used for computing the autocovariance, where it is assumed that
the differentials are uncorrelated beyond lags greater than `bandwidth`.

Autocovariances are smoothened by a cosine window, in order to decrease the variance of the
estimator.

For more details, see "Comparing Predictive Accuracy", F. X. Diebold, R. S. Mariano, Journal
of Business & Economic Statistics, (1995).
"""
function dm_mean_test(diffs; bandwidth::Integer=21)

    bandwidth > 0 || throw(DomainError(bandwidth, "Bandwidth must be a positive integer."))

    n = length(diffs)
    sample_mean = mean(diffs)
    lags = 0:bandwidth

    # Compute the effective sample size. First step is to get the autocovariance.
    # Since StatsBase implements the biased estimator (convolved with the triangular window)
    # the division is done to obtain the unbiased estimator.
    acov = autocov(diffs, lags) ./ autocov(ones(n), lags, demean=false)

    # Smoothen the spectral density with a cosine window.
    window = cos.(π * collect(lags) / (2 * bandwidth))
    acov .*= window

    # Get the spectral density at zero.
    # The first entry is the variance which should not receive a factor of 2.
    spec_dens = acov[1] + 2 * sum(acov[2:end])

    if spec_dens < 0
        throw(ArgumentError("Spectral density is negative: try using a smaller bandwidth."))
    end

    eff_size = sqrt(spec_dens / n)

    # Get test statistic
    Z = sample_mean / eff_size

    # Compute p value
    return 2 * cdf(Normal(), -abs(Z)) # Times 2 because this is a two-sided test.
end

ObservationDims.obs_arrangement(::typeof(dm_mean_test)) = MatrixColsOfObs()


"""
    dm_median_test(diffs; symmetric=false)

Compute the Diebold-Mariano p-value for the median of a series of loss differentials, `diffs`.

In practice, we take `diffs` as the difference between two financial return timeseries.
The null hypothesis is that the differentials have zero median.
!!!Note
   This is a different statement than saying the original losses have the same median.

Setting `symmetric=true` uses the Wilcoxon's Signed-Rank Test instead of a simple Sign-Test.
This should only be used if `diffs` is approximately symmetric, but brings extra power.

For more details, see "Comparing Predictive Accuracy", F. X. Diebold, R. S. Mariano, Journal
of Business & Economic Statistics, (1995).
"""
function dm_median_test(diffs; symmetric=false)

    n = length(diffs)

    if symmetric

        # Get ranks of the absolute differentials.
        ranks = sortperm(sortperm(abs.(diffs)))

        # Sum the ranks of the positive ones.
        tot = sum(ranks[diffs .> 0])

        # Compute test statistics.
        Z = (tot - n * (n + 1) / 4) / sqrt(n * (n + 1) * (2 * n + 1) / 24)

        # Compute p value.
        return 2 * cdf(Normal(), -abs(Z))
    else
        # Sum indicators of positive differentials.
        tot = sum(diffs .> 0)

        # Compute test statistics.
        Z = (tot - 0.5 * n) / sqrt(0.25 * n)

        # Compute p value.
        return 2 * cdf(Normal(), -abs(Z))
    end
end

ObservationDims.obs_arrangement(::typeof(dm_median_test)) = MatrixColsOfObs()
