@testset "evaluate.jl" begin
    orig_idist = KeyedDistribution(MvNormal(ones(3)), ["a", "b", "c"])

    # In these testsets the dummy_metric is simply a function that runs a set of tests.
    # The point is to verify that `evaluate` correctly rearrranges the (various) inputs
    # according to the ObsArrangement we specified for it.
    # It shows we can pass `evaluate` any number of various arguments and it will parse them
    # appropriately behind the scenes.

    @testset "SingleObs" begin
        function dummy_metric0(dist, idist, scalar, iter, mat)
            @test dist == Normal()
            @test idist == orig_idist
            @test scalar == 1
            @test iter == [1, 2, 3]
            @test collect(mat) == [10 20 30; 10 20 30]
        end
        ObservationDims.obs_arrangement(::typeof(dummy_metric0)) = SingleObs()
        evaluate(dummy_metric0, Normal(), orig_idist, 1, [1, 2, 3], [10 20 30; 10 20 30])
    end

    @testset "IteratorOfObs" begin
        function dummy_metric1(dist, idist, scalar, iter, mat)
            @test dist == Normal()
            @test idist == orig_idist
            @test scalar == 1
            @test iter == [1, 2, 3]
            @test collect(mat) == [[10, 20, 30], [10, 20, 30]]
        end
        ObservationDims.obs_arrangement(::typeof(dummy_metric1)) = IteratorOfObs()
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
        ObservationDims.obs_arrangement(::typeof(dummy_metric2)) = MatrixColsOfObs()
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
        ObservationDims.obs_arrangement(::typeof(dummy_metric3)) = MatrixRowsOfObs()
        evaluate(dummy_metric3, Normal(), orig_idist, 1, [1, 2, 3], [10 20 30; 10 20 30])
    end
end
