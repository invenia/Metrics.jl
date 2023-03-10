@testset "loglikelihood.jl" begin

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

            @testset "Using KeyedDistributions and KeyedArrays" begin

                rng = StableRNG(1)
                obs = ["a", "b", "c"]
                features = [:f1, :f2, :f3, :f4]
                id = KeyedDistribution(dist, obs)

                expected = marginal_gaussian_loglikelihood(y_true, dist)

                # normal
                a = KeyedArray(y_true; obs=obs, feature=features)
                @test marginal_gaussian_loglikelihood(a, id) ≈ expected

                #shuffled
                new_obs_order = shuffle(rng, 1:3)
                new_feature_order = shuffle(rng, 1:4)
                a = KeyedArray(
                    y_true[new_obs_order, new_feature_order],
                    obs=obs[new_obs_order],
                    feature=features[new_feature_order],
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

        rng = StableRNG(1)
        sqrtcov = rand(rng, 3, 3)
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

            @testset "Using KeyedDistributions with KeyedArrays" begin

                obs = ["a", "b", "c"]
                features = [:f1, :f2, :f3, :f4]
                id = KeyedDistribution(dist, obs)

                expected = joint_gaussian_loglikelihood(y_true, dist)

                # normal
                a = KeyedArray(y_true, obs=obs, feature=features)
                @test joint_gaussian_loglikelihood(a, id) ≈ expected

                #shuffled
                new_obs_order = shuffle(rng, 1:3)
                new_feature_order = shuffle(rng, 1:4)
                a = KeyedArray(
                    y_true[new_obs_order, new_feature_order],
                    obs=obs[new_obs_order],
                    feature=features[new_feature_order],
                )
                @test joint_gaussian_loglikelihood(a, id) ≈ expected

            end
        end
    end

    @testset "loglikelihood" begin
        rng = StableRNG(1)
        test_data = [
            # multivariate
            (
                dist_dim = "multivariate",
                pred_location = ones(3),
                pred_scale = [2.25 0.1 0.0; 0.1 2.25 0.0; 0.0 0.0 2.25],
            ),
        ]
        @testset "univariate" begin
            pred_location = 0.1
            pred_scale = 2
            y_true = 0.2
            @testset "Normal distribution" begin
                y_pred = Normal(pred_location, pred_scale)
                @test Metrics.loglikelihood(y_true, y_pred) ==
                    Distributions.loglikelihood(y_pred, y_true)
            end
            @testset "T distribution" begin
                y_pred = TDist(3.0)
                @test Metrics.loglikelihood(y_true, y_pred) ==
                    Distributions.loglikelihood(y_pred, y_true)
            end
        end

        @testset "multivariate" begin
            pred_location = ones(3)
            pred_scale = [2.25 0.1 0.0; 0.1 1.25 0.0; 0.0 0.0 3.25]
            y_true = [0.1, 0.2, 0.3]
            obs = Symbol.("t_", 1:3)
            y_true_idx = KeyedArray(y_true, obs=obs)
            @testset "Normal distribution" begin
                y_pred = MvNormal(pred_location, pred_scale)
                expected = Distributions.loglikelihood(y_pred, y_true)
                @test Metrics.loglikelihood(y_true, y_pred) == expected
                # - Using KeyedDistributions with KeyedArrays
                y_pred_idx = KeyedDistribution(y_pred, obs)
                @test Metrics.loglikelihood(y_true_idx, y_pred_idx) == expected
                # if the observation dimensions don't match the order, should use the axis
                # to match the order and give a correct resuilt
                new_obs_order = shuffle(rng, 1:3)
                y_true_idx2 = KeyedArray(y_true[new_obs_order], obs=obs[new_obs_order])
                @test Metrics.loglikelihood(y_true_idx2, y_pred_idx) == expected
            end
            @testset "T distribution" begin
                y_pred = Distributions.GenericMvTDist(3.0, pred_location, PDMat(pred_scale))
                expected = Distributions.loglikelihood(y_pred, y_true)
                @test Metrics.loglikelihood(y_true, y_pred) == expected
                # - Using KeyedDistributions with KeyedArrays
                y_pred_idx = KeyedDistribution(y_pred, obs)
                @test Metrics.loglikelihood(y_true_idx, y_pred_idx) == expected
                # if the observation dimensions don't match the order, should use the axis
                # to match the order and give a correct resuilt
                new_obs_order = shuffle(rng, 1:3)
                y_true_idx2 = KeyedArray(y_true[new_obs_order], obs=obs[new_obs_order])
                @test Metrics.loglikelihood(y_true_idx2, y_pred_idx) == expected
            end
        end
    end
end
