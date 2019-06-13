@testset "regression.jl" begin

    @testset "squared_error" begin
        @testset "scalar point" begin
            y_true = 1
            y_pred = 1
            @test squared_error(y_true, y_pred) == 0
            y_true = 4
            y_pred = 1
            @test squared_error(y_true, y_pred) == 9
            y_true = rand(Int64)
            y_pred = rand(Int64)
            @test squared_error(y_true, y_pred) == squared_error(y_pred, y_true)
        end
        @testset "vector point" begin
            y_true = [1, 2, 3]
            y_pred = [1, 2, 3]
            @test squared_error(y_true, y_pred) == 0
            y_true = [2, 2, 2, 2]
            y_pred = [1, 2, 3, 4]
            @test squared_error(y_true, y_pred) == 6
            y_true = rand(Int64, 7)
            y_pred = rand(Int64, 7)
            @test squared_error(y_true, y_pred) == squared_error(y_pred, y_true)
        end
        @testset "erroring" begin
            y_true = 1
            y_pred = [1, 2, 3]
            @test_throws DimensionMismatch squared_error(y_true, y_pred)
            y_true = [2, 2]
            y_pred = [1, 2, 3, 4]
            @test_throws DimensionMismatch squared_error(y_true, y_pred)
        end
    end

    @testset "marginal_loglikelihood" begin
        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_pred = [.1, .2, .3]
            @test marginal_loglikelihood(dist, y_pred) == -4.842507141293854
        end
        @testset "vector point" begin
            dist = MvNormal(3, 2)
            y_pred = [8 10 10; 10 5 7; 9 7 10]
            @test marginal_loglikelihood(dist, y_pred) == -98.00877142388157
        end
    end

    @testset "joint_loglikelihood" begin
        @testset "scalar point" begin
            dist = Normal(0.1, 2)
            y_pred = [.1, .2, .3]
            @test joint_loglikelihood(dist, y_pred) == -4.842507141293854
            @test joint_loglikelihood(dist, y_pred) == marginal_loglikelihood(dist, y_pred)
        end
        @testset "vector point" begin
            dist = MvNormal(3, 2)
            y_pred = [8 10 10; 10 5 7; 9 7 10]
            @test joint_loglikelihood(dist, y_pred) == -98.00877142388157
        end
    end

    @testset "evaluate" begin
        @testset "squared_error" begin
            dist = rand()
            y_pred = rand()
            @test evaluate(squared_error, dist, y_pred) == squared_error(dist, y_pred)
        end
    end

end
