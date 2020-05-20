@testset "summaries.jl" begin
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
    end
end
