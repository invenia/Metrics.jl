"""
    kullback_leibler(a::MvNormal, b::MvNormal) -> Float64
    kullback_leibler(a::Normal, b::Normal) -> Float64

Calculate the Kullback-Leibler divergence between two gaussian distributions.

https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
"""
function kullback_leibler(a::MvNormal, b::MvNormal)
    @_dimcheck size(a) == size(b)

    # Pull out the covariance matrices
    Σ0, Σ1 = cov(a), cov(b)

    # Make sure that the covariance is not singular
    d0, d1 = det(Σ0), det(Σ1)
    d0 != 0 || throw(ArgumentError("Covariance matrix Σ0 is singular"))
    d1 != 0 || throw(ArgumentError("Covariance matrix Σ1 is singular"))

    # k is the Distribution Dimension
    k = length(a)

    μ0 = mean(a)
    μ1 = mean(b)
    μdiff = μ1 .- μ0

    # μdiff' * (Σ1 \ μdiff) can be more efficiently computed using the Cholesky
    Z = cholesky(Σ1).L \ μdiff

    # Calculate the Kullback Leibler Divergence
    kl = 0.5 * (tr(Σ1 \ Σ0) + Z' * Z - k + log(d1 / d0))

    return kl
end

function kullback_leibler(a::Normal, b::Normal)
    # Convert the Univariate Distributions to 1D Multivariate Distributions
    mv_a = MvNormal([mean(a)], std(a))
    mv_b = MvNormal([mean(b)], std(b))
    return kullback_leibler(mv_a, mv_b)
end

"""
    kullback_leibler(a::KeyedMvNormal, b::KeyedMvNormal) -> Float64

Calculate the Kullback-Leibler divergence between two `KeyedMvNormal` distributions.
"""
function kullback_leibler(a::KeyedMvNormal, b::KeyedMvNormal)
    @_dimcheck size(a) == size(b)

    id1, d1 = only(axiskeys(a)), distribution(a)
    id2, d2 = only(axiskeys(b)), distribution(b)

    if !(sort(id1) == sort(id2))
        throw(ArgumentError(
            "Distribution axiskeys do not match: "*
            "axiskeys(a) = ($id1,), axiskeys(b) = ($id2,)",
        ))
    end

    p1 = sortperm(id1)
    p2 = sortperm(id2)

    sorted_d1 = MvNormal(d1.μ[p1], cov(d1)[p1, p1])
    sorted_d2 = MvNormal(d2.μ[p2], cov(d2)[p2, p2])

    return kullback_leibler(sorted_d1, sorted_d2)
end

ObservationDims.obs_arrangement(::typeof(kullback_leibler)) = SingleObs()
const kl = kullback_leibler
