@testset "summaries.jl" begin
    FD = FixedDecimal{Int, 2}
    @testset "Regression Summaries" begin
        # We don't need to check the exact values returned, we already have robust tests for each metric.
        # but we do want to make sure the summaries come back in a type we can consume.
        # Technically we just need to check we are getting a NamedTuple with right fields here
        # but checking the types gives a bit more confirmation.
        function generate_expected_type(y_true, y_pred)
            isscalar = y_true isa Number && y_pred isa Number
            add_val = first(first(y_true .+ y_pred))
            div_val = add_val/add_val

            example = (;
                :mean_squared_error => isscalar ? add_val : div_val,
                :root_mean_squared_error => div_val,
                :normalised_root_mean_squared_error => div_val,
                :standardized_mean_squared_error => div_val,
                :mean_absolute_error => isscalar ? add_val : div_val,
                :potential_payoff => div_val
            )
            return typeof(example)
        end

        @testset "Scalar" begin
            y_true = [1,2,3,4]
            y_pred = [5,6,7,8]

            expected_type = generate_expected_type(first(y_true), first(y_pred))
            summary = regression_summary(first(y_true), first(y_pred))

            @test !isempty(summary)
            @test summary isa expected_type

            expected_type = generate_expected_type(y_true, y_pred)
            summary = evaluate(regression_summary, y_true, y_pred)

            @test !isempty(summary)
            @test summary isa expected_type
        end

        @testset "Scalar - Indexed Values" begin
            y_true = [1,2,3,4]
            y_pred = [5,6,7,8]

            expected_type = generate_expected_type(y_true, y_pred)
            summary = regression_summary(y_true, y_pred)

            @test !isempty(summary)
            @test summary isa expected_type
        end

        @testset "Vector" begin
            y_true = [[1,2,3], [4,5,6], [7,8,9]]
            y_pred = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

            expected_type = generate_expected_type(first(y_true), first(y_pred))
            summary = regression_summary(first(y_true), first(y_pred))

            @test !isempty(summary)
            @test summary isa expected_type

            expected_type = generate_expected_type(y_true, y_pred)
            summary = evaluate(regression_summary, y_true, y_pred)

            @test !isempty(summary)
            @test summary isa expected_type
        end

        @testset "Vector - Indexed Values" begin
            y_true = [[1,2,3], [4,5,6], [7,8,9]]
            y_pred = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

            expected_type = generate_expected_type(y_true, y_pred)
            summary = regression_summary(y_true, y_pred)

            @test !isempty(summary)
            @test summary isa expected_type
        end

        @testset "Matrix" begin
            y_true = [[1 2; 3 4], [3 4; 5 6], [5 6; 7 8]]
            y_pred = [[7 8; 9 10], [9 10; 11 12], [11 12; 13 14]]

            expected_type = generate_expected_type(first(y_true), first(y_pred))
            summary = regression_summary(first(y_true), first(y_pred))

            @test !isempty(summary)
            @test summary isa expected_type

            expected_type = generate_expected_type(y_true, y_pred)
            summary = evaluate(regression_summary, y_true, y_pred)

            @test !isempty(summary)
            @test summary isa expected_type
        end

        @testset "Matrix - Indexed Values" begin
            y_true = [[1 2; 3 4], [3 4; 5 6], [5 6; 7 8]]
            y_pred = [[7 8; 9 10], [9 10; 11 12], [11 12; 13 14]]

            expected_type = generate_expected_type(y_true, y_pred)
            summary = regression_summary(y_true, y_pred)

            @test !isempty(summary)
            @test summary isa expected_type
        end

        @testset "EIS use case" begin
            # Normally, `y_true` is a `Vector{<:Real}`, `y_pred` is a multivariate distribution.
            # Typical distribution type includes `MvNormal` and `GenericMvTDist`.

            # Choose the `regression_metrics` list in the following way as they are the main
            # metrics we checked on regression performance.
            regression_metrics = [Metrics.loglikelihood, mse2m]
            # y_pred
            pred_location = ones(3)
            pred_scale = [2.25 0.1 0.0; 0.1 1.25 0.0; 0.0 0.0 3.25]
            y_pred_list = [
                (dtype="MvN", y_pred=MvNormal(pred_location, pred_scale)),
                (dtype="MvT", y_pred=Distributions.GenericMvTDist(3.0, pred_location, PDMat(pred_scale)))
            ]
            # y_true
            y_true = [6, 3, 5]
            obs = Symbol.("t_", 1:3)
            y_true_idx = KeyedArray(y_true; obs=obs)
            @testset "distribution type: $(i.dtype)" for i in y_pred_list
                summary = regression_summary(y_true, i.y_pred)
                @test !isempty(summary)
                # KeyedArray with KeyedDistribution
                y_pred_idx = KeyedDistribution(i.y_pred, obs)
                summary_withidx = regression_summary(y_true_idx, y_pred_idx)
                @test summary_withidx == summary
                # suffle the keys
                new_obs_order = shuffle(1:3)
                y_true_idx2 = KeyedArray(y_true[new_obs_order]; obs=obs[new_obs_order])
                summary_withdiffidx = regression_summary(y_true_idx2, y_pred_idx)
                @test summary_withdiffidx == summary
            end
        end
    end
    @testset "Financial Summaries" begin
        @testset "volumes::AbstractVector, deltas::Union{MvNormal, AbstractMatrix}" begin
            num_nodes = 20
            volumes = rand(Uniform(-50,50), num_nodes)
            mean_deltas = rand(num_nodes)
            deltas = generate_mvnormal(mean_deltas, num_nodes)

            expected_type = typeof((;
                :expected_return => 1.0,
                :expected_shortfall => 1.0,
                :sharpe_ratio => 1.0,
                :volatility => 1.0,
            ))

            summary = financial_summary(volumes, deltas)

            @test summary isa expected_type

            @testset "per MWh" begin
                summary_mwh = financial_summary(volumes, deltas; per_mwh=true)
                total_volume = sum(abs, volumes)

                @test summary_mwh.expected_return == summary.expected_return / total_volume
                @test summary_mwh.expected_shortfall == summary.expected_shortfall / total_volume
                @test summary_mwh.sharpe_ratio == summary.sharpe_ratio
                @test summary_mwh.volatility == summary.volatility / total_volume
            end
        end

        @testset "returns::AbstractVector; risk_level::Real=0.5" begin
            returns = rand(20)
            risk_level = 0.05

            expected_type = typeof((;
                :median_return => 1.0,
                :expected_return => 1.0,
                :expected_shortfall => 1.0,
                :expected_windfall => 1.0,
                :median_over_expected_shortfall => 1.0,
                :sharpe_ratio => 1.0,
                :volatility => 1.0,
                :mean_over_expected_shortfall => 1.0,
            ))

            summary = financial_summary(returns; risk_level=risk_level)

            @test summary isa expected_type

            @testset "returns iterator" begin
                # Bonus from this test since the summary calls all other things
                # this ensures they all accept an iterator
                summary = financial_summary(skipmissing(returns); risk_level=risk_level)

                # None are missing so doing `skipmissing` will not change anything
                @test summary isa expected_type
            end
        end

            # Tests for the financial_summary function.
        @testset "returns, volumes; kwargs..." begin

            # The keys we expect in our dictionary
            # Sort them becaue ordering doesn't matter and it makes comparison easier.
            expected_keys = sort([
                :total_volume,
                :mean_volume,
                :median_volume,
                :std_volume,
                :total_return,
                :mean_return,
                :median_return,
                :std_return,
                :expected_shortfall,
                :expected_windfall,
                :mean_over_expected_shortfall,
                :median_over_expected_shortfall,
            ])

            @testset "weird inputs" begin
                # empty input
                test_returns = Vector{Int}()
                test_volumes = Vector{Int}()

                result = financial_summary(test_returns, test_volumes)

                @test sort(collect(keys(result))) == expected_keys

                @test isequal(result[:total_volume], missing)
                @test isequal(result[:mean_volume], missing)
                @test isequal(result[:median_volume], missing)
                @test isequal(result[:std_volume], missing)

                @test isequal(result[:total_return], missing)
                @test isequal(result[:mean_return], missing)
                @test isequal(result[:median_return], missing)
                @test isequal(result[:std_return], missing)

                @test isequal(result[:expected_shortfall], missing)

                @test isequal(result[:mean_over_expected_shortfall], missing)
                @test isequal(result[:median_over_expected_shortfall], missing)


                # Empty Input but the type is FixedDecimal, which doesn't have a NaN type
                test_returns = Vector{FD}()
                test_volumes = Vector{FD}()

                result = financial_summary(test_returns, test_volumes)

                @test sort(collect(keys(result)))  == expected_keys

                @test isequal(result[:total_volume], missing)
                @test isequal(result[:mean_volume], missing)
                @test isequal(result[:median_volume], missing)
                @test isequal(result[:std_volume], missing)

                @test isequal(result[:total_return], missing)
                @test isequal(result[:mean_return], missing)
                @test isequal(result[:median_return], missing)
                @test isequal(result[:std_return], missing)

                @test isequal(result[:expected_shortfall], missing)

                @test isequal(result[:mean_over_expected_shortfall], missing)
                @test isequal(result[:median_over_expected_shortfall], missing)

            end

            @testset "different lengths" begin
                # Test input with no losses
                test_returns = [1, 2, 3, 4, 5, 6]
                test_volumes = [1, 2, 3, 4, 5, 6, 7, 9]

                @test_throws ErrorException financial_summary(test_returns, test_volumes, risk_level=0.5)
            end

            @testset "NaN return/volume" begin
                test_returns = [0, -1, 2]
                test_volumes = [0, 1, 2]

                for per_mwh in [true, false]

                    # es percentile to 50 so that es metrics are
                    # not 'missing' (!isnan(missing) == missing)
                    result = financial_summary(test_returns, test_volumes,
                                            risk_level=0.5, per_mwh=per_mwh)

                    for val in values(result)
                        @test !isnan(val)
                    end
                end

            end

            # Tests with simple expected inputs to test basic functionality
            @testset "simple input" begin
                # Test input with no losses
                test_returns = [1, 2, 3, 4, 5, 6]
                test_volumes = [1, 2, 3, 4, 5, 6]

                result = financial_summary(test_returns, test_volumes, risk_level=0.2)

                @test sort(collect(keys(result))) == expected_keys

                @test isequal(result[:total_volume], 21)
                @test isequal(result[:mean_volume], 3.5)
                @test isequal(result[:median_volume], 3.5)
                @test isequal(result[:std_volume], 1.8708286933869707)

                @test isequal(result[:total_return], 21)
                @test isequal(result[:mean_return], 3.5)
                @test isequal(result[:median_return], 3.5)
                @test isequal(result[:std_return], 1.8708286933869707)


                # ES uses bottom 5% by default, so for test_returns this is just the worst return.
                @test isequal(result[:expected_shortfall], -1.0)
                @test isequal(result[:mean_over_expected_shortfall], 3.5 / -1.0)
                @test isequal(result[:median_over_expected_shortfall], 3.5 / -1.0)


                # Test input with only losses
                test_returns = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
                test_volumes = [1, 2, 3, 4, 5, 6]

                result = financial_summary(test_returns, test_volumes, risk_level=0.2)

                @test sort(collect(keys(result)))  == expected_keys

                @test isequal(result[:total_volume], 21)
                @test isequal(result[:mean_volume], 3.5)
                @test isequal(result[:median_volume], 3.5)
                @test isequal(result[:std_volume], 1.8708286933869707)

                @test isequal(result[:total_return], -21)
                @test isequal(result[:mean_return], -3.5)
                @test isequal(result[:median_return], -3.5)
                @test isequal(result[:std_return], 1.8708286933869707)

                # ES uses bottom 5% by default, so for test_returns this is just the worst return.
                @test isequal(result[:expected_shortfall], 6.0)
                @test isequal(result[:mean_over_expected_shortfall], -3.5 / 6.0)
                @test isequal(result[:median_over_expected_shortfall], -3.5 / 6.0)


                # Test input with some losses
                test_returns = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
                test_volumes = [1, 2, 3, 4, 5, 6]

                result = financial_summary(test_returns, test_volumes, risk_level=0.2)

                @test sort(collect(keys(result)))  == expected_keys

                @test isequal(result[:total_volume], 21)
                @test isequal(result[:mean_volume], 3.5)
                @test isequal(result[:median_volume], 3.5)
                @test isequal(result[:std_volume], 1.8708286933869707)

                @test isequal(result[:total_return], -3)
                @test isequal(result[:mean_return], -0.5)
                @test isequal(result[:median_return], -0.5)
                @test isequal(result[:std_return], 4.230839160261236)

                # ES uses bottom 5% by default, so for test_returns this is just the worst return.
                @test isequal(result[:expected_shortfall], 6.0)
                @test isequal(result[:mean_over_expected_shortfall], -0.5 / 6.0)
                @test isequal(result[:median_over_expected_shortfall], -0.5 / 6.0)


                # Test with a larger ES percentage
                test_returns = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
                test_volumes = [1, 2, 3, 4, 5, 6]
                test_es_percentile = 0.5

                result = financial_summary(
                    test_returns,
                    test_volumes;
                    risk_level=test_es_percentile
                )

                @test sort(collect(keys(result)))  == expected_keys

                @test isequal(result[:total_volume], 21)
                @test isequal(result[:mean_volume], 3.5)
                @test isequal(result[:median_volume], 3.5)
                @test isequal(result[:std_volume], 1.8708286933869707)

                @test isequal(result[:total_return], -3)
                @test isequal(result[:mean_return], -0.5)
                @test isequal(result[:median_return], -0.5)
                @test isequal(result[:std_return], 4.230839160261236)

                @test isequal(result[:expected_shortfall], 4)
                @test isequal(result[:mean_over_expected_shortfall], -0.125)
                @test isequal(result[:median_over_expected_shortfall], -0.125)

            end

            # Test volumes being negative
            @testset "negative volumes" begin
                # A few negative volumes we expect to be absolute valued
                test_returns = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
                test_volumes = [-1, -2, -3, 4, 5, 6]
                test_unchanged_volumes = [-1, -2, -3, 4, 5, 6]
                test_es_percentile = 0.5

                result = financial_summary(
                    test_returns,
                    test_volumes;
                    risk_level=test_es_percentile
                )

                @test sort(collect(keys(result)))  == expected_keys

                @test isequal(result[:total_volume], 21)
                @test isequal(result[:mean_volume], 3.5)
                @test isequal(result[:median_volume], 3.5)
                @test isequal(result[:std_volume], 1.8708286933869707)

                @test isequal(result[:total_return], -3)
                @test isequal(result[:mean_return], -0.5)
                @test isequal(result[:median_return], -0.5)
                @test isequal(result[:std_return], 4.230839160261236)

                @test isequal(result[:expected_shortfall], 4)
                @test isequal(result[:mean_over_expected_shortfall], -0.125)
                @test isequal(result[:median_over_expected_shortfall], -0.125)

                # Test that the volumes didn't get changed by calling the function.
                @test test_volumes == test_unchanged_volumes
            end


            @testset "FixedDecimal" begin
                # Test input that would give a missing for some fields
                test_returns = [FD(1), FD(2), FD(3), FD(4), FD(5), FD(6)]
                test_volumes = [FD(1), FD(2), FD(3), FD(4), FD(5), FD(6)]

                result = financial_summary(test_returns, test_volumes, risk_level=0.2)

                @test sort(collect(keys(result)))  == expected_keys

                @test isequal(result[:total_volume], 21)
                @test isequal(result[:mean_volume], 3.5)
                @test isequal(result[:median_volume], 3.5)
                @test isequal(result[:std_volume], 1.8708286933869707)

                @test isequal(result[:total_return], 21)
                @test isequal(result[:mean_return], 3.5)
                @test isequal(result[:median_return], 3.5)
                @test isequal(result[:std_return], 1.8708286933869707)

                @test isequal(result[:expected_shortfall], FD(-1))
                @test isequal(result[:mean_over_expected_shortfall], 3.5 / -FD(1))
                @test isequal(result[:median_over_expected_shortfall], 3.5 / -FD(1))

            end


            # Tests what happens when the inputs have "missing" values
            @testset "Missing values" begin
                # Test only missing values.
                test_returns = [missing, missing, missing]
                test_volumes = [missing, missing, missing]

                result = financial_summary(test_returns, test_volumes)

                # Test that the dict has exactly the keys we expect.
                @test sort(collect(keys(result)))  == expected_keys

                # Test that every key has the expected value.
                @test isequal(result[:total_volume], missing)
                @test isequal(result[:mean_volume], missing)
                @test isequal(result[:median_volume], missing)
                @test isequal(result[:std_volume], missing)

                @test isequal(result[:total_return], missing)
                @test isequal(result[:mean_return], missing)
                @test isequal(result[:median_return], missing)
                @test isequal(result[:std_return], missing)

                @test isequal(result[:expected_shortfall], missing)

                @test isequal(result[:mean_over_expected_shortfall], missing)
                @test isequal(result[:median_over_expected_shortfall], missing)

                # Test some missing values added to a previous dataset
                # A few negative volumes we expect to be absolute valued
                test_returns = [1.0, missing, -2.0, 3.0, missing, -4.0, 5.0, missing, -6.0, missing]
                test_volumes = [-1, missing, -2, -3, missing, 4, 5, missing, 6, missing]

                test_unchanged_volumes = [-1, missing, -2, -3, missing, 4, 5, missing, 6, missing]
                test_es_percentile = 0.5

                result = financial_summary(
                    test_returns,
                    test_volumes;
                    risk_level=test_es_percentile
                )

                # Test that every key has the expected value.
                @test sort(collect(keys(result)))  == expected_keys

                @test isequal(result[:total_volume], 21)
                @test isequal(result[:mean_volume], 3.5)
                @test isequal(result[:median_volume], 3.5)
                @test isequal(result[:std_volume], 1.8708286933869707)

                @test isequal(result[:total_return], -3)
                @test isequal(result[:mean_return], -0.5)
                @test isequal(result[:median_return], -0.5)
                @test isequal(result[:std_return], 4.230839160261236)

                @test isequal(result[:expected_shortfall], 4)

                @test isequal(result[:mean_over_expected_shortfall], -0.125)
                @test isequal(result[:median_over_expected_shortfall], -0.125)

                # Test that the volumes didn't get changed by calling the function.
                @test isequal(test_volumes, test_unchanged_volumes)


                # Test with missings in different places in the array
                test_returns = [1.0, -2.0, missing, 3.0, -4.0, 5.0, -6.0, missing, missing, missing]
                test_volumes = [-1, missing, -2, -3, missing, 4, 5, missing, 6, missing]
                # These array should look like this after missings are removed:
                # [1, 3, 5, -6]
                # [-1, -3, 4, 5]

                test_unchanged_volumes = [-1, missing, -2, -3, missing, 4, 5, missing, 6, missing]
                test_es_percentile = 0.5

                result = financial_summary(
                    test_returns,
                    test_volumes;
                    risk_level=test_es_percentile
                )

                # Test that every key has the expected value.
                @test sort(collect(keys(result)))  == expected_keys

                @test result[:total_volume] == 13
                @test result[:mean_volume] == 3.25
                @test result[:median_volume] == 3.5
                @test result[:std_volume] == 1.707825127659933

                @test result[:total_return] == 3
                @test result[:mean_return] == 0.75
                @test result[:median_return] == 2.0
                @test result[:std_return] == 4.7871355387816905

                @test result[:expected_shortfall] == 2.5

                @test result[:mean_over_expected_shortfall] == 0.3
                @test result[:median_over_expected_shortfall] == 0.8

                # Test that the volumes didn't get changed by calling the function.
                @test isequal(test_volumes, test_unchanged_volumes)
            end
        end
    end
end
