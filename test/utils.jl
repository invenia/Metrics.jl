using Metrics: @_dimcheck, _match
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

    @testset "_match" begin

        plain_array = [3, 2, 1]
        plain_dist = generate_mvnormal([2, 1, 3], 3)

        axis_array = AxisArray(plain_array, Axis{:obs}(["b", "a", "c"]))
        index_dist = IndexedDistribution(plain_dist, ["c", "b", "a"])

        @testset "Plain array and plain distribution" begin
            new_array, new_dist = _match(plain_array, plain_dist)
            @test new_array == plain_array
            @test new_dist == plain_dist
        end

        @testset "AxisArray and plain distribution" begin
            new_array, new_dist = _match(axis_array, plain_dist)
            @test new_array == axis_array
            @test new_dist == plain_dist
        end

        @testset "Plain array and IndexedDistribution" begin
            new_array, new_dist = _match(plain_array, index_dist)
            @test new_array == plain_array
            @test new_dist == index_dist
        end

        @testset "AxisArray and IndexedDistribution" begin
            new_array, new_dist = _match(axis_array, index_dist)
            @test new_array != axis_array
            @test new_dist != index_dist

            # check indices are now sorted and values are in new order
            @test index(new_dist) == axisvalues(new_array)[1] == ["a", "b", "c"]
            @test mean(parent(new_dist)) == [3, 1, 2]
            @test parent(new_array) == [2, 3, 1]
        end

        @testset "AxisArray and AxisArray" begin
            axis_array2 = AxisArray([3, 1, 2], Axis{:obs}(["c", "a", "b"]))
            new_array1, new_array2 = _match(axis_array, axis_array2)
            @test new_array1 != axis_array
            @test new_array2 != axis_array2

            # check indices are now sorted and values are in new order
            @test axisvalues(new_array1)[1] == axisvalues(new_array2)[1] == ["a", "b", "c"]
            @test parent(new_array1) == [2, 3, 1]
            @test parent(new_array2) == [1, 2, 3]
        end
    end
end
