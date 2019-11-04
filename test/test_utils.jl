# helper function to generate MvNormals
function generate_mvnormal(mean, size::Integer)
    X = rand(size, size)
    rand_cov = X' * X + 0.01I
    return MvNormal(mean, Symmetric(rand_cov))
end
generate_mvnormal(size::Integer) = generate_mvnormal(rand(size), size)

# relocate a distribution to a new mean
relocate(d::Normal, new_μ) = Normal(new_μ, d.σ)
relocate(d::MvNormal, new_μ) = MvNormal(new_μ, d.Σ)
relocate(d::MatrixNormal, new_μ) = MatrixNormal(new_μ, d.U, d.V)

# broadcast sensibly if passed a vector of distributions
relocate(v::AbstractVector{<:Distribution}, new_mu) = relocate.(v, new_mu)

# rescale a distribution variance by scale_factor
rescale(d::Normal, scale_factor) = Normal(d.μ, scale_factor * d.σ)
rescale(d::MvNormal, scale_factor) = MvNormal(d.μ, PSDMat(scale_factor * cov(d)))

function rescale(d::MatrixNormal, U_sf::Number, V_sf::Number)
    return MatrixNormal(d.M, PSDMat(U_sf * d.U.mat), PSDMat(V_sf * d.V.mat))
end
rescale(d::MatrixNormal, U_sf) = rescale(d::MatrixNormal, U_sf, U_sf)

# broadcast sensibly if passed a vector of distributions
rescale(v::AbstractVector{<:Distribution}, scale_factor) = rescale.(v, Ref(scale_factor))

# return the distribution with unit variance
unit_variance(d::Normal) = Normal(d.μ, 1)
unit_variance(d::MvNormal) = MvNormal(d.μ, 1)
unit_variance(d::MatrixNormal) = MatrixNormal(d.M, zeros(size(d.U)...) + I, zeros(size(d.V)...) + I)

# This mean function returns the point-predictions from vector of distributions.
Statistics.mean(y_pred::AbstractVector{<:Sampleable}) = mean.(y_pred)
