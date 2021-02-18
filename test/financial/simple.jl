using Metrics: split_volume

@testset "simple financial metrics" begin
    @testset "expected return" begin
        @testset "basic properties" begin
            # simple use of base function
            volumes = [0, 1, -10, 100]
            deltas = [0.1, 0.2, 0.3, 0.4]
            expected = 37.2
            @test expected_return(volumes, deltas) == expected
            @test evaluate(expected_return, volumes, deltas) == expected

            # negating both input should make no difference
            @test isequal(
                expected_return(volumes, deltas),
                expected_return(-volumes, -deltas)
            )
            # negating one input should flip the sign on the result
            @test isequal(
                expected_return(volumes, -deltas),
                expected_return(-volumes, deltas)
            )
            @test isequal(
                expected_return(volumes, deltas),
                -expected_return(-volumes, deltas)
            )

            # with price impact
            nonzero_pi = (supply_pi=fill(0.1, 4), demand_pi=fill(0.1, 4))
            supply, demand = split_volume(volumes)
            pi = dot(nonzero_pi.supply_pi, supply.^2) + dot(nonzero_pi.demand_pi, demand.^2)

            # update previous expected result
            expected -= pi
            @test expected_return(volumes, deltas, nonzero_pi...) == expected
            @test expected < expected_return(volumes, deltas)

        end

        # using dense cov matrix
        num_nodes = 20
        volumes = rand(Uniform(-50,50), num_nodes)
        mean_deltas = rand(num_nodes)
        _sqrt = rand(num_nodes, num_nodes+2)
        scale_deltas = PDMat(Symmetric(_sqrt * _sqrt' + I))

        nonzero_pi = (supply_pi=fill(0.1, num_nodes), demand_pi=fill(0.1, num_nodes))

        node_names = "node" .* string.(collect(1:num_nodes))

        @testset "distribution type $(typeof(dense_dist))" for dense_dist in [
            MvNormal(mean_deltas, scale_deltas),
            GenericMvTDist(2.2, mean_deltas, scale_deltas)
        ]

            @testset "with $type" for (type, dist) in (
                ("Distribution", dense_dist),
                ("IndexedDistribution", IndexedDistribution(dense_dist, node_names))
            )
                expected = dot(volumes, mean_deltas)
                @test expected_return(volumes, dist) ≈ expected
                @test evaluate(expected_return, volumes, dist) ≈ expected

                # with price impact
                @test expected_return(volumes, dist, nonzero_pi...) < expected
                @test evaluate(expected_return, volumes, dist, nonzero_pi...) < expected
            end

            @testset "with samples" begin
                # using sample deltas
                samples = rand(dense_dist, 10)
                expected = dot(volumes, mean(samples, dims=2))
                @test expected_return(volumes, samples) ≈ expected
                @test evaluate(expected_return, volumes, samples; obsdim=2) ≈ expected

                @test expected_return(volumes, samples, nonzero_pi...) < expected
                @test isless(
                    evaluate(expected_return, volumes, samples, nonzero_pi...; obsdim=2),
                    expected,
                )
            end
        end

        @testset "AbstractVector" begin
            sample = [1,2,3,4,5]
            expected = 3.0

            @test expected_return(sample) == expected
        end

        @testset "Single value" begin
            @test expected_return(500) == 500
        end

        @testset "Empty vector - MethodError" begin
            @test_throws MethodError expected_return([])
        end
    end

    @testset "volatility" begin
        # using diagonal cov matrix
        diag_sqrtcov = Diagonal([5.0, 6.0 ,7.0])
        diag_dist = MvNormal(rand(3), diag_sqrtcov' * diag_sqrtcov)
        @test isequal(
            volatility([0.1, 1.0, -10.0], diag_dist),
            sqrt(0.5^2 + 6^2 + 70^2)
        )

        # using dense cov matrix
        num_nodes = 10
        volumes = rand(Uniform(-50,50), num_nodes)
        mean_deltas = rand(num_nodes)
        _sqrt = rand(num_nodes, num_nodes+2)
        scale_deltas = PDMat(Symmetric(_sqrt * _sqrt' + I))

        nonzero_pi = (supply_pi=fill(0.1, num_nodes), demand_pi=fill(0.1, num_nodes))

        node_names = "node" .* string.(collect(1:num_nodes))

        @testset "distribution type $(typeof(dense_dist))" for dense_dist in [
            MvNormal(mean_deltas, scale_deltas),
            # use a reasonably big dof for MvT, otherwise the empircal volatility error would be big
            GenericMvTDist(10, mean_deltas, scale_deltas)
        ]
            @testset "with $type" for (type, dist) in (
                ("Distribution", dense_dist),
                ("IndexedDistribution", IndexedDistribution(dense_dist, node_names))
            )
                expected = norm(sqrtcov(StatsUtils.cov(dist)) * volumes, 2)
                @test volatility(volumes, dist) ≈ expected
                @test evaluate(volatility, volumes, dist) ≈ expected

                # test against empirical results
                samples = rand(dist, 1_000_000)
                empirical_vol = std(samples' * volumes)
                @test volatility(volumes, dist) ≈ empirical_vol atol=1e-1
            end

            @testset "with samples" begin
                samples = rand(dense_dist, 5)
                expected = std(samples' * volumes)
                @test volatility(volumes, samples) ≈ expected
                @test evaluate(volatility, volumes, samples; obsdim=2) ≈ expected
            end
        end

        @testset "AbstractVector" begin
            sample = [1,2,3,4,5]
            expected = 1.5811388300841898

            @test volatility(sample) ≈ expected
        end

        @testset "Single value" begin
            @test isnan(volatility(0))
        end

        @testset "Empty vector - MethodError" begin
            @test_throws MethodError volatility([])
        end
    end

    @testset "sharpe ratio" begin
        @testset "simple sharpe ratio" begin
            @testset "vector of returns" begin
                # basic usage
                returns = rand(100)
                mean_returns = mean(returns)
                std_returns = std(returns)
                expected = mean_returns / std_returns

                @test sharpe_ratio(returns) ≈ expected
                @test evaluate(sharpe_ratio, returns) ≈ expected

                # order shouldn't matter
                shuffle!(returns)
                @test sharpe_ratio(returns) ≈ expected
                @test evaluate(sharpe_ratio, returns) ≈ expected

                # zero std with non-zero mean
                returns = ones(10)
                @test sharpe_ratio(returns) == Inf
                @test evaluate(sharpe_ratio, returns) ≈ Inf

                # zero std with zero mean
                returns = zeros(10)
                @test isnan(sharpe_ratio(returns))
                @test isnan(evaluate(sharpe_ratio, returns))
            end

            @testset "distribution of returns" begin
                returns = Normal(rand(), rand())
                mean_returns = mean(returns)
                std_returns = std(returns)
                expected = mean_returns / std_returns

                @test sharpe_ratio(returns) == expected
                @test evaluate(sharpe_ratio, returns) == expected

                # zero std with non-zero mean
                returns = Normal(rand(), 0)
                @test sharpe_ratio(returns) == Inf
                @test evaluate(sharpe_ratio, returns) == Inf

                # zero std with zero mean
                returns = Normal(0, 0)
                @test isnan(sharpe_ratio(returns))
                @test isnan(evaluate(sharpe_ratio, returns))

            end
        end

        @testset "using volumes and deltas" begin
            # using diag cov matrix
            vol = [0.1, 0.2, -0.3]
            diag_sqrtcov = Diagonal([5.0, 6.0 ,7.0])
            diag_dist = MvNormal([0.0, 1.0, 10.0], diag_sqrtcov' * diag_sqrtcov)
            exp_return = (0.2 + -3.0)
            exp_vol = sqrt(0.5^2 + 1.2^2 + 2.1^2)
            @test sharpe_ratio(vol, diag_dist) ≈ exp_return / exp_vol

            # using dense cov matrix
            volumes = rand(Uniform(-50,50), 10)
            dense_dist = generate_mvnormal(10)
            nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))

            names = "nodes" .* string.(collect(1:10))
            dense_id = IndexedDistribution(dense_dist, names)

            @testset "with $type" for (type, dist) in (
                ("Distribution", dense_dist),
                ("IndexedDistribution", dense_id)
            )
                exp_return = expected_return(volumes, dist)
                exp_vol = volatility(volumes, dist)

                @test sharpe_ratio(volumes, dist) == exp_return / exp_vol
                @test evaluate(sharpe_ratio, volumes, dist) == exp_return / exp_vol

                # with price impact
                @test sharpe_ratio(volumes, dist, nonzero_pi...) < exp_return / exp_vol
                @test isless(
                    evaluate(sharpe_ratio, volumes, dist, nonzero_pi...),
                    exp_return / exp_vol,
                )
            end

            @testset "with samples" begin
                samples = rand(dense_dist, 5)

                exp_return = expected_return(volumes, samples)
                exp_vol = volatility(volumes, samples)

                @test sharpe_ratio(volumes, samples) == exp_return / exp_vol
                @test evaluate(sharpe_ratio, volumes, samples; obsdim=2) ≈ exp_return / exp_vol

                # with price impact sharpe ratio should decrease
                @test sharpe_ratio(volumes, samples, nonzero_pi...) < exp_return / exp_vol
                @test isless(
                    evaluate(sharpe_ratio, volumes, samples, nonzero_pi...; obsdim=2),
                    exp_return / exp_vol,
                )
            end
        end
    end

    @testset "median_return" begin
        @testset "volumes and deltas" begin
            # Basic Usage
            volumes = collect(1:10)
            deltas = Matrix(I, (10, 10))
            expected = 5.5

            @test median_return(volumes, deltas) == expected
            @test evaluate(median_return, volumes, deltas) == expected

            # With Price Impact median_return should be lower
            nonzero_pi = (supply_pi=fill(0.1, 10), demand_pi=fill(0.1, 10))
            pi_expected = -33

            @test median_return(volumes, deltas, nonzero_pi...) == pi_expected
            @test evaluate(median_return, volumes, deltas, nonzero_pi...) == pi_expected
            @test pi_expected < expected
        end
        @testset "returns" begin
            @test median_return([1.2, 3, 4, 5, 6, 6, 9]) == 5
        end
    end
end
