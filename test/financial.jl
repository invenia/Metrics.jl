using Metrics: @_dimcheck
@testset "financial.jl" begin
    @testset "expected_shortfall" begin
        @testset "normal usage" begin
            returns = collect(1:100)
            α = .05
            @test expected_shortfall(returns, α) == 3
            α = .25
            @test expected_shortfall(returns, α) == 13
            returns = [3, 5, 7, 8, 10, 11, 11, 12, 14, 15, 15, 16, 16, 17, 17, 17]
            shuffle!(returns)
            α = 1/16
            @test expected_shortfall(returns, α) == 3
            α = 3.5/16
            @test expected_shortfall(returns, α) == 5
            α = rand(1/16:1/32:15/16)
            @test expected_shortfall(returns, α) == expected_shortfall(sort(returns), α)
            α = 4/16
            returns_2 = sort(returns)
            returns_2[5:end] .= 1000
            shuffle!(returns_2)
            @test expected_shortfall(returns, α) == expected_shortfall(returns_2, α)
            α = 5/16
            @test expected_shortfall(returns, α) != expected_shortfall(returns_2, α)
        end
        @testset "erroring" begin
            returns = collect(1:100)
            α = 1
            @test_throws MethodError expected_shortfall(returns, α)
            α = 0
            @test_throws MethodError expected_shortfall(returns, α)
            α = 1.
            @test_throws ArgumentError expected_shortfall(returns, α)
            α = 0.
            @test_throws ArgumentError expected_shortfall(returns, α)
            α = -.5
            @test_throws ArgumentError expected_shortfall(returns, α)
            α = 1.1
            @test_throws ArgumentError expected_shortfall(returns, α)

            returns = []
            α = 1/2
            @test_throws ArgumentError expected_shortfall(returns, α)
            returns = [1]
            @test_throws ArgumentError expected_shortfall(returns, α)
            returns = collect(1:100)
            α = 1/101
            @test_throws ArgumentError expected_shortfall(returns, α)

        end
    end

end
