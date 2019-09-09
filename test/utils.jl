using Metrics: @_dimcheck
@testset "utils.jl" begin

    @testset "failing checks" begin
        @test_throws DimensionMismatch @_dimcheck 1 == 3

        try
            @_dimcheck 1 == 3
        catch err
            @test err isa DimensionMismatch
            @test occursin("1 == 3", sprint(showerror, err))
        end

        try
            @_dimcheck size([20,30]) == size([10,20,30])
        catch err
            @test err isa DimensionMismatch
            @test occursin("(2,) == (3,)", sprint(showerror, err))
        end

        try
            a = [1, 2]
            b = [1, 2, 3]
            @_dimcheck a == b
        catch err
            @test err isa DimensionMismatch
            @test occursin("a == b", sprint(showerror, err))
        end

        try
            @_dimcheck size([20,30]) == size([10,20,30]) == size([10,20,30])
        catch err
            @test err isa DimensionMismatch
            @test occursin("(2,)==(3,)==(3,)", sprint(showerror, err))
        end

        try
            @_dimcheck size(1,2) == 1 == "foo"
        catch err
            @test err isa DimensionMismatch
            @test occursin("1==1==foo", sprint(showerror, err))
        end
end

    @testset "passing checks" begin
        @test @_dimcheck(1===1) isa Any
        @test @_dimcheck(size([20,30]) == (2,)) isa Any
        @test @_dimcheck(size([20,30]) == size([2,3]) == (2,)) isa Any
    end

end
