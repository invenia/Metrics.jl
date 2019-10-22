@testset "organise_obs" begin
    c_organise_obs(args...; kwargs...) = collect(organise_obs(args...; kwargs...))

    @testset "SingleObs" begin

        data = (
            1.2,
            [1, 2, 3],
            [[1, 2, 3], [4, 5, 6]],
            [[1 2 3; 10 20 30], [4 5 6; 40 50 60]],
            Normal(1, 3),
            MvNormal(rand(3), I)
        )

        # SingleObs should basically do nothing to these data
        @test all(==(raw,organise_obs(SingleObs(), raw)) for raw in data)

        data = (
            [1, 2, 3],
            [[1, 2, 3], [4, 5, 6]],
            [[1 2 3; 10 20 30], [4 5 6; 40 50 60]],
        )

        for expected in data
            # add an extra artificial column of size 1 to the end, e.g. (3,) -> (3, 1)
            # organise_obs should remove this
            with_extra_col = reshape(expected, (size(expected)..., :))
            @test expected == organise_obs(SingleObs(), with_extra_col)

            # add 2 extra columns of size 1 at random, e.g. (3,) -> (1, 3, 1)
            # organise_obs should also remove these
            new_size = shuffle([size(expected)..., 1, 1])
            with_extra_cols = reshape(expected, (new_size...))
            @test expected == organise_obs(SingleObs(), with_extra_col)

        end

    end

    @testset "simple IteratorOfObs for iterators" begin
        for out in (
            [1,2,3],
            [[1,2,3], [4,5,6]],
            [[1 2 3; 10 20 30], [4 5 6; 40 50 60]],
        )
            for transform in (identity, Tuple, x->Base.Generator(identity, x))
                raw = transform(out)
                @test out == c_organise_obs(IteratorOfObs(), raw)
            end
        end
    end

    @testset "iterators to $Arrange" for (Arrange, out) in (
        (MatrixRowsOfObs(),  [10 20 30; 40 50 60]),
        (MatrixColsOfObs(), [10 40; 20 50; 30 60]),
    )
        raw = [[10, 20, 30], [40, 50, 60]]

        for transform in (identity, Tuple, x->Base.Generator(identity, x))
            data_iter = transform(raw)
            @test out == c_organise_obs(Arrange, data_iter)
        end
    end


    @testset "matrix to $Arrange" for (Arrange, out) in (
        (IteratorOfObs(), [[1,2,3],[4,5,6]]),
        (MatrixRowsOfObs(), [1 2 3; 4 5 6]),
        (MatrixColsOfObs(), [1 4; 2 5; 3 6]),
    )
        raw = [1 2 3; 4 5 6]
        @test out == c_organise_obs(Arrange, raw)  # default is rows
        @test out == c_organise_obs(Arrange, raw; obsdim=1)
        @test out == c_organise_obs(Arrange, raw'; obsdim=2)
        @test out == c_organise_obs(Arrange, NamedDimsArray{(:obs, :var)}(raw))
        @test out == c_organise_obs(Arrange, NamedDimsArray{(:var, :obs)}(raw'))
        @test out == c_organise_obs(Arrange, NamedDimsArray{(:x, :y)}(raw'); obsdim=:y)
        @test out == c_organise_obs(Arrange, NamedDimsArray{(:x, :y)}(raw); obsdim=:x)
    end
end

@testset "how evaluate handles mixs" begin
    orig_idist = IndexedDistribution(MvNormal(ones(3)), ["a", "b", "c"])

    @testset "IteratorOfObs" begin
        function dummy_metric1(dist, idist, scalar, iter, mat)
            @test dist == Normal()
            @test idist == orig_idist
            @test scalar == 1
            @test iter == [1, 2, 3]
            @test collect(mat) == [[10, 20, 30], [10, 20, 30]]
        end
        Metrics.obs_arrangement(::typeof(dummy_metric1)) = IteratorOfObs()
        evaluate(dummy_metric1, Normal(), orig_idist, 1, [1, 2, 3], [10 20 30; 10 20 30])
    end

    @testset "MatrixColsOfObs" begin
        function dummy_metric2(dist, idist, scalar, iter, mat)
            @test dist == Normal()
            @test idist == orig_idist
            @test scalar == 1
            @test iter == [1, 2, 3]
            @test collect(mat) == [10 10;  20 20; 30 30]
        end
        Metrics.obs_arrangement(::typeof(dummy_metric2)) = MatrixColsOfObs()
        evaluate(dummy_metric2, Normal(), orig_idist, 1, [1, 2, 3], [10 20 30; 10 20 30])
    end

    @testset "MatrixRowsOfObs" begin
        function dummy_metric3(dist, idist, scalar, iter, mat)
            @test dist == Normal()
            @test idist == orig_idist
            @test scalar == 1
            @test iter == [1, 2, 3]
            @test collect(mat) == [10 20 30; 10 20 30]
        end
        Metrics.obs_arrangement(::typeof(dummy_metric3)) = MatrixRowsOfObs()
        evaluate(dummy_metric3, Normal(), orig_idist, 1, [1, 2, 3], [10 20 30; 10 20 30])
    end
end
