"""
    kullback_leibler(
        a::Distribution{Multivariate},
        b::Distribution{Multivariate}
    ) -> Float64

Calculate the Kullback-Leibler divergence between two gaussian distributions.

https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
"""
function kullback_leibler(a::Distribution{Multivariate}, b::Distribution{Multivariate})
    @_dimcheck size(a) == size(b)
    Σ0 = cov(a)
    Σ1 = cov(b)
    k = length(a) # Matrix Dimension
    𝜇0 = mean(a)
    𝜇1 = mean(b)
    𝜇diff = 𝜇1 - 𝜇0
    kl = (1 / 2) * (
        tr(inv(Σ1) * Σ0) + transpose(𝜇diff) * inv(Σ1) * 𝜇diff - k + log(det(Σ1) / det(Σ0))
    )
    return kl
end

obs_arrangement(::typeof(kullback_leibler)) = MatrisColsofObs()
const kl = kullback_leibler
