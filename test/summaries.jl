@testset "summaries.jl" begin
    @testset "Regression Summaries" begin
        function generate_expected_values(y_true, y_pred)
            expected = Dict()

            for metric in REGRESSION_METRICS
                expected[metric] = metric(y_true, y_pred)
            end

            return expected
        end

        @testset "Scalar" begin
            y_true = [1,2,3,4]
            y_pred = [5,6,7,8]

            expected = generate_expected_values(first(y_true), first(y_pred))
            summary = regression_summary(first(y_true), first(y_pred))

            @test !isempty(summary)
            @test isequal(expected, summary)

            expected = generate_expected_values(y_true, y_pred)
            summary = evaluate(regression_summary, y_true, y_pred)

            @test !isempty(summary)
            @test isequal(expected, summary)
        end

        @testset "Scalar - Indexed Values" begin
            y_true = [1,2,3,4]
            y_pred = [5,6,7,8]

            expected = generate_expected_values(y_true, y_pred)
            summary = regression_summary(y_true, y_pred)

            @test !isempty(summary)
            @test isequal(expected[mse], summary[mse])
            @test isequal(expected[rmse], summary[rmse])
            @test isequal(expected[nrmse], summary[nrmse])
            @test isequal(expected[smse], summary[smse])
        end

        @testset "Vector" begin
            y_true = [[1,2,3], [4,5,6], [7,8,9]]
            y_pred = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

            expected = generate_expected_values(first(y_true), first(y_pred))
            summary = regression_summary(first(y_true), first(y_pred))

            @test !isempty(summary)
            @test isequal(expected, summary)

            expected = generate_expected_values(y_true, y_pred)
            summary = evaluate(regression_summary, y_true, y_pred)

            @test !isempty(summary)
            @test isequal(expected, summary)
        end

        @testset "Vector - Indexed Values" begin
            y_true = [[1,2,3], [4,5,6], [7,8,9]]
            y_pred = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

            expected = generate_expected_values(y_true, y_pred)
            summary = regression_summary(y_true, y_pred)

            @test !isempty(summary)
            @test isequal(expected[mse], summary[mse])
            @test isequal(expected[rmse], summary[rmse])
            @test isequal(expected[nrmse], summary[nrmse])
            @test isequal(expected[smse], summary[smse])
        end

        @testset "Matrix" begin
            y_true = [[1 2; 3 4], [3 4; 5 6], [5 6; 7 8]]
            y_pred = [[7 8; 9 10], [9 10; 11 12], [11 12; 13 14]]

            expected = generate_expected_values(first(y_true), first(y_pred))
            summary = regression_summary(first(y_true), first(y_pred))

            @test !isempty(summary)
            @test isequal(expected, summary)

            expected = generate_expected_values(y_true, y_pred)
            summary = evaluate(regression_summary, y_true, y_pred)

            @test !isempty(summary)
            @test isequal(expected, summary)
        end

        @testset "Matrix - Indexed Values" begin
            y_true = [[1 2; 3 4], [3 4; 5 6], [5 6; 7 8]]
            y_pred = [[7 8; 9 10], [9 10; 11 12], [11 12; 13 14]]

            expected = generate_expected_values(y_true, y_pred)
            summary = regression_summary(y_true, y_pred)

            @test !isempty(summary)
            @test isequal(expected[mse], summary[mse])
            @test isequal(expected[rmse], summary[rmse])
            @test isequal(expected[nrmse], summary[nrmse])
            @test isequal(expected[smse], summary[smse])
        end
    end
    @testset "Financial Summaries" begin
        @testset "volumes::AbstractVector, deltas::Union{MvNormal, AbstractMatrix}" begin
            num_nodes = 20
            volumes = rand(Uniform(-50,50), num_nodes)
            mean_deltas = rand(num_nodes)
            deltas = generate_mvnormal(mean_deltas, num_nodes)

            expected = Dict()
            expected[expected_return] = expected_return(volumes, deltas)
            expected[expected_shortfall] = expected_shortfall(volumes, deltas)
            expected[sharpe_ratio] = sharpe_ratio(volumes, deltas)
            expected[volatility] = volatility(volumes, deltas)

            summary = financial_summary(volumes, deltas)

            @test isequal(expected, summary)
       end

        @testset "returns::AbstractVector; risk_level::Real=0.5" begin
            returns = rand(20)
            risk_level = 0.05

            expected = Dict()
            expected[expected_return] = expected_return(returns)
            expected[median_return] = median_return(returns)
            expected[expected_shortfall] = expected_shortfall(returns; risk_level=risk_level)
            expected[median_over_expected_shortfall] =
                median_over_expected_shortfall(returns; risk_level=risk_level)
            expected[sharpe_ratio] = sharpe_ratio(returns)
            expected[volatility] = volatility(returns)

            summary = financial_summary(returns; risk_level=risk_level)

            @test isequal(expected, summary)

            @testset "returns iterator" begin
                # Bonus from this test since the summary calls all other things
                # this ensures they all accept an iterator
                summary = financial_summary(skipmissing(returns); risk_level=risk_level)

                # None are missing so doing `skipmissing` will not change anything
                @test isequal(expected, summary)
            end
        end
    end
end
