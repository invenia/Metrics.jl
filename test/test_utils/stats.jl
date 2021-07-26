# helper function to generate MvNormals
function generate_mvnormal(mean, size::Integer)
    @assert length(mean) == size
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


@testset "Test Utils" begin
    @testset "generate_mvnormal" begin

        m = [1, 2, 3]
        d = generate_mvnormal(m, 3)
        @test d isa Distribution{<:Multivariate}
        @test size(d) == (3,)
        @test mean(d) == m

        m = 1.2
        @test_throws AssertionError generate_mvnormal(m, 4)

        d = generate_mvnormal(2)
        @test d isa Distribution{<:Multivariate}
        @test size(d) == (2,)

    end

    @testset "relocate" begin

        @testset "Normal" begin
            d1 = Normal()

            new_mean = 2.7
            d2 = relocate(d1, new_mean)
            @test mean(d2) == new_mean
            @test std(d1) == std(d2)
        end
        @testset "MvNormal" begin
            d1 = MvNormal([1.2, -0.3], 2)

            new_mean = [1, -1]
            d2 = relocate(d1, new_mean)
            @test mean(d2) == new_mean
            @test Matrix(cov(d1)) == Matrix(cov(d2))

        end
        @testset "MatrixNormal" begin

            U = [1 2; 2 4.5]
            V = [1 2 3; 2 5.5 10.2; 3 10.2 24]
            M = [1 2 3; 4 5 6]
            d1 = MatrixNormal(M, U, V)

            new_mean = [1.4 -2 -0.1; 4 3.1 -3.6]
            d2 = relocate(d1, new_mean)
            @test mean(d2) == new_mean
            @test d1.U == d2.U
            @test d1.V == d2.V
        end
    end

    @testset "rescale" begin

        @testset "Normal" begin
            d1 = Normal(0.1, 2.3)

            f = 0.7
            d2 = rescale(d1, f)
            @test mean(d1) == mean(d2)
            @test std(d2) == f * std(d1)

            f = 0
            d2 = rescale(d1, f)
            @test mean(d1) == mean(d2)
            @test std(d2) == f * std(d1) == 0

        end

        @testset "MvNormal" begin

            Σ = [2 1 1; 1 2.2 2; 1 2 3]
            μ = [7, 6, 5]
            d1 = MvNormal(μ, Σ)

            f = 0.2
            d2 = rescale(d1, f)
            @test mean(d1) == mean(d2)
            @test Matrix(cov(d2)) == f .* Matrix(cov(d1))

            f = 0
            d2 = rescale(d1, f)
            @test mean(d1) == mean(d2)
            @test Matrix(cov(d2)) == f .* Matrix(cov(d1)) == zeros(size(Σ)...)

        end

        @testset "MatrixNormal" begin

            U = [1 2; 2 4.5]
            V = [1 2 3; 2 5.5 10.2; 3 10.2 24]
            M = [1 2 3; 4 5 6]
            d1 = MatrixNormal(M, U, V)

            f1, f2 = 1.3, 0.2
            d2 = rescale(d1, f1, f2)
            @test mean(d1) == mean(d2)
            @test d2.U.mat == f1 .* d1.U.mat
            @test d2.V.mat == f2 .* d1.V.mat

            f = 0.2
            d2 = rescale(d1, f)
            @test mean(d1) == mean(d2)
            @test d2.U.mat == f .* d1.U.mat
            @test d2.V.mat == f .* d1.V.mat

            f = 0.0
            d2 = rescale(d1, f)
            @test mean(d1) == mean(d2)
            @test d2.U.mat == zeros(size(d1.U)...)
            @test d2.V.mat == zeros(size(d1.V)...)

        end

    end

    @testset "unit_variance" begin

        @testset "Normal" begin
            d1 = Normal(2, 3)

            d2 = unit_variance(d1)
            @test mean(d1) == mean(d2)
            @test std(d2) == 1
        end

        @testset "MvNormal" begin
            Σ = [2 1 1; 1 2.2 2; 1 2 3]
            μ = [7, 6, 5]
            d1 = MvNormal(μ, Σ)

            d2 = unit_variance(d1)
            @test mean(d1) == mean(d2)
            @test Matrix(cov(d2)) == [1 0 0; 0 1 0; 0 0 1]

        end

        @testset "MatrixNormal" begin
            U = [1 2; 2 4.5]
            V = [1 2 3; 2 5.5 10.2; 3 10.2 24]
            M = [1 2 3; 4 5 6]
            d1 = MatrixNormal(M, U, V)

            d2 = unit_variance(d1)
            @test mean(d1) == mean(d2)
            @test d2.U.mat == [1 0; 0 1]
            @test d2.V.mat == [1 0 0; 0 1 0; 0 0 1]
        end
    end

    @testset "mean on Vector{<:Sampleable}" begin

        @testset "Normal" begin
            means = rand(5)
            vars = [1, 2, 3, 4, 5]
            dists = Normal.(means, vars)

            @test mean(dists) == means
        end

        @testset "MvNormal" begin
            means = [rand(3), rand(3), rand(3)]
            sigmas = [1, 2, 3]
            dists = MvNormal.(means, sigmas)

            @test mean(dists) == means
        end

        @testset "MatrixNormal" begin
            U = [1 2; 2 4.5]
            V = [1 2 3; 2 5.5 10.2; 3 10.2 24]
            means = [rand(2, 3), rand(2, 3), rand(2, 3)]
            dists = MatrixNormal.(means, Ref(U), Ref(V))

            @test mean(dists) == means
        end

        @testset "KeyedDistribution" begin
            means = [rand(5), rand(5), rand(5)]
            sigmas = [1, 2, 3]
            dists = MvNormal.(means, sigmas)

            names = ["a", "b", "c", "d", "e"]
            ind_dists = KeyedDistribution.(dists, Ref(names))

            expected = KeyedArray.(means, Ref(names))
            @test mean(ind_dists) == expected
        end
    end
end
