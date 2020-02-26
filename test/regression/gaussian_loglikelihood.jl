@testset "gaussian_loglikelihood.jl" begin

    @testset "marginal_gaussian_loglikelihood" begin

        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_true = [0.1, 0.2, 0.3]
            y_mean = [0.1, 0.1, 0.1]

            @testset "Properties" begin
                @test marginal_gaussian_loglikelihood(y_true, dist) < 0.0 # logprobs always negative
                # it's a symmetric metric (e.g. the order of the two input arguments can change)
                marginal_gaussian_loglikelihood(y_true, dist) == marginal_gaussian_loglikelihood(dist, y_true)
                # y_true is less likely than y_mean
                @test marginal_gaussian_loglikelihood(y_true, dist) < marginal_gaussian_loglikelihood(y_mean, dist)
            end
        end

        @testset "vector point" begin
            dist = MvNormal(3, 1.5)
            y_true = [
                8.  10   9  11
                10   5   7  12
                10   7  10  1
            ]
            y_mean = zeros(3, 4)

            @testset "Properties" begin
                @test marginal_gaussian_loglikelihood(y_true, dist) < 0.0 # logprobs always negative
                # it's a symmetric metric (e.g. the order of the two input arguments can change)
                marginal_gaussian_loglikelihood(y_true, dist) == marginal_gaussian_loglikelihood(dist, y_true)
                # y_true is less likely than y_mean
                @test marginal_gaussian_loglikelihood(y_true, dist) < marginal_gaussian_loglikelihood(y_mean, dist)
                # using the alternative Canonical form should not change results
                @test marginal_gaussian_loglikelihood(y_true, dist) ≈ marginal_gaussian_loglikelihood(y_true, canonform(dist))
            end

            @testset "Using IndexedDistributions and AxisArrays" begin

                obs = ["a", "b", "c"]
                features = [:f1, :f2, :f3, :f4]
                id = IndexedDistribution(dist, obs)

                expected = marginal_gaussian_loglikelihood(y_true, dist)

                # normal
                a = AxisArray(y_true, Axis{:obs}(obs), Axis{:feature}(features))
                @test marginal_gaussian_loglikelihood(a, id) ≈ expected

                #shuffled
                new_obs_order = shuffle(1:3)
                new_feature_order = shuffle(1:4)
                a = AxisArray(
                    y_true[new_obs_order, new_feature_order],
                    Axis{:obs}(obs[new_obs_order]),
                    Axis{:feature}(features[new_feature_order]),
                )
                @test marginal_gaussian_loglikelihood(a, id) ≈ expected

            end
        end
    end

    @testset "joint_gaussian_loglikelihood" begin

        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_true = [.1, .2, .3]
            y_mean = [0.1, 0.1, 0.1]

            @testset "Properties" begin
                @test joint_gaussian_loglikelihood(y_true, dist) < 0.0  # logprobs always negative
                # it's a symmetric metric (e.g. the order of the two input arguments can change)
                joint_gaussian_loglikelihood(y_true, dist) == joint_gaussian_loglikelihood(dist, y_true)
                # y_true is less likely than y_mean
                @test joint_gaussian_loglikelihood(y_true, dist) < joint_gaussian_loglikelihood(dist, y_mean)
                # For unviariate markingal and joint are the same, it is just the normalized likelyhood.
                @test joint_gaussian_loglikelihood(y_true, dist) ≈ marginal_gaussian_loglikelihood(y_true, dist)
            end
        end

        sqrtcov = rand(3, 3)
        @testset "vector point" for dist in (MvNormal(3, 1.5), MvNormal(zeros(3), sqrtcov*sqrtcov'))
            y_true = [
                8.  10   9  11
                10   5   7  12
                10   7  10  1
            ]
            y_mean = zeros(3, 4)

            @testset "Properties" begin
                @test joint_gaussian_loglikelihood(y_true, dist) < 0.0  # logprobs always negative
                # it's a symmetric metric (e.g. the order of the two input arguments can change)
                joint_gaussian_loglikelihood(y_true, dist) == joint_gaussian_loglikelihood(dist, y_true)
                # y_true is less likely than y_mean
                @test joint_gaussian_loglikelihood(y_true, dist) < joint_gaussian_loglikelihood(dist, y_mean)

                if dist isa ZeroMeanIsoNormal
                    # For IsoNormal joint and marginal are the same, it is just the normalized likelyhood.
                    @test joint_gaussian_loglikelihood(y_true, dist) ≈ marginal_gaussian_loglikelihood(y_true, dist)
                else
                    @test joint_gaussian_loglikelihood(y_true, dist) != marginal_gaussian_loglikelihood(y_true, dist)
                end

                # using the alternative canonical form should not change the results
                @test joint_gaussian_loglikelihood(y_true, dist) ≈ joint_gaussian_loglikelihood(y_true, canonform(dist))

            end

            @testset "Using IndexedDistributions with AxisArrays" begin

                obs = ["a", "b", "c"]
                features = [:f1, :f2, :f3, :f4]
                id = IndexedDistribution(dist, obs)

                expected = joint_gaussian_loglikelihood(y_true, dist)

                # normal
                a = AxisArray(y_true, Axis{:obs}(obs), Axis{:feature}(features))
                @test joint_gaussian_loglikelihood(a, id) ≈ expected

                #shuffled
                new_obs_order = shuffle(1:3)
                new_feature_order = shuffle(1:4)
                a = AxisArray(
                    y_true[new_obs_order, new_feature_order],
                    Axis{:obs}(obs[new_obs_order]),
                    Axis{:feature}(features[new_feature_order]),
                )
                @test joint_gaussian_loglikelihood(a, id) ≈ expected

            end
        end
    end
end
