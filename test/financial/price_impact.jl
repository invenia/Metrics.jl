using Metrics: split_volume

@testset "split_volume" begin
    mixed = [0, 1, -2, 3, -4, 5, -6]
    pos, neg = split_volume(mixed)
    @test all(pos .>= 0)
    @test all(neg .<= 0)
end


@testset "price impact" begin
    rng = StableRNG(1)
    N = 50  # e.g. number of nodes
    nonzero_pi = (supply_pi=fill(0.1, N), demand_pi=fill(0.1, N))

    # Ensure supply is positive and demand negative,
    # no nodes have both supply and demand,
    # some nodes have zero volume (so we test this case).
    volumes = rand(rng, Uniform(-20, 20), N)
    zero_vols = sample(1:N, 4, replace = false)
    volumes[zero_vols] .= 0

    supply, demand = split_volume(volumes)

    @testset "PI empty" begin
        @test price_impact(volumes) == 0
    end

    @testset "PI matrix" begin
        # identity price impact matrix
        Pi = Matrix(I, N, N)
        @test price_impact(volumes, Pi) == dot(volumes, volumes)

        # generic price impact matrix
        Pi = [1.0 -2.3; 5.2 0.9]
        vol = [0.9, 1.7]
        @test price_impact(vol, Pi) ≈ 7.848
    end

    @testset "PI vectors" begin
        # test Pi = 0
        @test price_impact(volumes, zeros(N), zeros(N)) == 0
        # Test Pi != 0 with volumes
        @test price_impact(volumes, nonzero_pi...) > 0
        # Test Pi != with supply, demand
        @test price_impact(supply, demand, nonzero_pi...) > 0
        # both methods should give same result
        @test price_impact(volumes, nonzero_pi...) == price_impact(supply, demand, nonzero_pi...)
    end

    @testset "erroring" begin
        # wrong type for volume
        @test_throws MethodError price_impact(1.2, rand(2,2))
        @test_throws MethodError price_impact(3.4, [1, 1], [2, 2])

        # wrong length
        @test_throws DimensionMismatch price_impact([1, 2, 3], [1, 2], [1, 1], [2, 2])
    end
end
