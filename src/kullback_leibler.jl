"""
    kullback_leibler(
        a::Distribution{Multivariate},
        b::Distribution{Multivariate}
    ) -> Float64

Calculate the Kullback-Leibler divergence between two gaussian distributions.

https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
"""
function kullback_leibler(a::MvNormal, b::MvNormal)
    @_dimcheck size(a) == size(b)
    @_dimcheck length(a) == length(b)

    # Pull out the covariance matrices
    Σ0 = cov(a)
    Σ1 = cov(b)

    # Make sure that the covariance is not singular
    @_dimcheck det(Σ0) != 0
    @_dimcheck det(Σ1) != 0

    # k is the Distribution Dimension
    k = length(a)

    # Pull out the means
    𝜇0 = mean(a)
    𝜇1 = mean(b)
    𝜇diff = 𝜇1 .- 𝜇0

    # 𝜇diff' * (Σ1 \ 𝜇diff) can be more efficiently computed using the Cholesky
    Z = cholesky(Σ1).L \ 𝜇diff

    kl = 0.5 * (tr(Σ1 \ Σ0) + Z' * Z - k + log(det(Σ1) / det(Σ0)))

    return kl
end

obs_arrangement(::typeof(kullback_leibler)) = SingleObs()
const kl = kullback_leibler
