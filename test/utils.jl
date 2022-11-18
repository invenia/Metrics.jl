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

        keyed_array = KeyedArray(plain_array; obs=["b", "a", "c"])
        keyed_dist = KeyedDistribution(plain_dist, ["c", "b", "a"])

        @testset "Plain array and plain distribution" begin
            new_array, new_dist = _match(plain_array, plain_dist)
            @test new_array == plain_array
            @test new_dist == plain_dist
        end

        @testset "KeyedArray and plain distribution" begin
            new_array, new_dist = _match(keyed_array, plain_dist)
            @test new_array == keyed_array
            @test new_dist == plain_dist
        end

        @testset "Plain array and KeyedDistribution" begin
            new_array, new_dist = _match(plain_array, keyed_dist)
            @test new_array == plain_array
            @test new_dist == keyed_dist
        end

        @testset "KeyedArray and KeyedDistribution" begin
            new_array, new_dist = _match(keyed_array, keyed_dist)
            @test new_array != keyed_array
            @test new_dist == keyed_dist

            # check the axiskeys are sorted to match the distribution keys, while
            # the order of the distribution keys remains the same
            @test only(axiskeys(new_dist)) == only(axiskeys(new_array)) == only(axiskeys(keyed_dist))
            @test mean(distribution(new_dist)) == [2, 1, 3]
            @test parent(new_array) == [1, 3, 2]
        end

        @testset "KeyedArray and KeyedArray" begin
            rng = StableRNG(1)
            @testset "1-D" begin
                keyed_array2 = KeyedArray([3, 1, 2]; obs=["c", "a", "b"])
                new_array1, new_array2 = _match(keyed_array, keyed_array2)
                @test new_array1 != keyed_array
                @test new_array2 != keyed_array2

                # check indices are now sorted and values are in new order
                @test only(axiskeys(new_array1)) == only(axiskeys(new_array2)) == ["a", "b", "c"]
                @test parent(new_array1) == [2, 3, 1]
                @test parent(new_array2) == [1, 2, 3]
            end
            @testset "2-D" begin
                keyed_array1 = KeyedArray(
                    [1 3; 2 -1; 5 0];
                    obs=["c", "a", "b"],
                    target=[:t1, :t2]
                )
                keyed_array2 = KeyedArray(
                    [2 0; 3 5; -2 6];
                    obs=["a", "c", "b"],
                    target=[:t2, :t1],
                )
                new_array1, new_array2 = _match(keyed_array1, keyed_array2)
                @test new_array1 != keyed_array1
                @test new_array2 != keyed_array2

                # check indices are now sorted and values are in new order
                @test axiskeys(new_array1)[1] == axiskeys(new_array2)[1] == ["a", "b", "c"]
                @test axiskeys(new_array1)[2] == axiskeys(new_array2)[2] == Symbol[:t1, :t2]
                @test parent(new_array1) == [2 -1; 5 0; 1 3]
                @test parent(new_array2) == [0 2; 6 -2; 5 3]
            end

            @testset "array axes have wrong orientation" begin
                a = KeyedArray(rand(rng, 3,2); obs=["c", "a", "b"], target=[:t1, :t2])
                b = KeyedArray(rand(rng, 3,2); target=["c", "a", "b"], obs=[:t1, :t2])
                @test_throws ArgumentError _match(a, b)
            end

            @testset "array axis values do not match" begin
                a = KeyedArray(rand(rng, 3,2); obs=["c", "a", "b"], target=[:t1, :t2])
                b = KeyedArray(rand(rng, 3,2); obs=["q", "a", "b"], target=[:t1, :t2])
                @test_throws ArgumentError _match(a, b)
            end

            @testset "arrays have incompatible sizes" begin
                a = KeyedArray(rand(rng, 3,2); obs=["c", "a", "b"], target=[:t1, :t2])
                b = KeyedArray(rand(rng, 2,3); target=[:t1, :t2], obs=["c", "a", "b"])
                @test_throws DimensionMismatch _match(a, b)
            end
        end
    end
end
