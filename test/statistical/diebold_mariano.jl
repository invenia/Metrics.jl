@testset "diebold_mariano" begin

    seed!(1)

    @testset "dm_mean" begin

        @testset "basic test" begin
            expected = 0.5
            result = mean(dm_mean_test.(eachslice(randn(10000, 10000), dims=2)))
            @test isapprox(result, expected, rtol=0.05)
        end

        @testset "detects signficant mean differences" begin
            result = map([0, 0.001, 0.1, 1.0, 10.0]) do m
                data = rand(Normal(m, 1), 10000)
                dm_mean_test(data)
            end

            @test (result .< 0.05) == [false, false, true, true, true]
        end

        @testset "detects significance over certain sample size" begin
            result = map([100, 500, 1_000, 5_000, 10_000]) do s
                data = rand(Normal(0.1, 1), s)
                dm_mean_test(data)
            end

            @test sort(result, rev=true) == result
            @test (result .< 0.05) == [false, false, true, true, true]
        end


        @testset "p-value decays beyond critical bandwidth" begin

            # create a 2 time-series with lag 10
            s = collect(0.01:0.01:1) + rand(Normal(0, 1), 100);
            x1 = s[1:end-10]
            x2 = s[11:end]

            result = map([5, 10, 15, 20]) do b
                dm_mean_test(x1-x2; bandwidth=b)
            end

            # the p-value should slightly peak when lag = bandwidth
            @test result[1] < result[2]

            # beyond the critical bandwidth the p-values should decay
            @test result[2] > result[3] > result[4]

        end

        @testset "bandwidth is not a positive integer" begin
            @test_throws DomainError dm_mean_test(100; bandwidth=0)
            @test_throws DomainError dm_mean_test(100; bandwidth=-1)
            @test_throws TypeError dm_mean_test(100; bandwidth=1.1)
        end

        @testset "bandwidth is too large" begin
            @test_throws ErrorException dm_mean_test(1:100; bandwidth=101)
        end

        @testset "spectral density is negative" begin
            @test_throws ArgumentError dm_mean_test(rand(100); bandwidth=80)
        end

    end

    @testset "dm_median" begin

        @testset "basic test" begin

            data = eachslice(randn(1000, 1000); dims=2)
            expected = 0.5

            @testset "symmetric" begin
                result = mean(dm_median_test.(data; symmetric=true))
                @test isapprox(result, expected, rtol=0.05)
            end

            @testset "non-symmetric" begin
                result = mean(dm_median_test.(data; symmetric=false))
                @test isapprox(result, expected, rtol=0.05)
            end
        end

        @testset "detects signficance median differences" begin

            means = [0, 0.001, 0.1, 1.0, 10.0]

            @testset "symmetric" begin
                result_symm = map(means) do m
                    data = rand(Normal(m, 1), 1000)
                    dm_median_test(data; symmetric=true)
                end

                @test (result_symm .< 0.05) == [false, false, true, true, true]
            end

            @testset "non-symmetric" begin
                result_nonsymm = map(means) do m
                    data = rand(Normal(m, 1), 1000)
                    dm_median_test(data; symmetric=false)
                end

                @test (result_nonsymm .< 0.05) == [false, false, true, true, true]
            end

        end

        @testset "detects signficance over certain sample size" begin

            sizes = [100, 500, 1_000]

            @testset "symmetric" begin
                result_symm = map(sizes) do s
                    data = rand(Normal(0.2, 1), s)
                    dm_median_test(data; symmetric=true)
                end

                @test (result_symm .< 0.05) == [false, true, true]
            end

            @testset "non-symmetric" begin
                result_nonsymm = map(sizes) do s
                    data = rand(Normal(0.2, 1), s)
                    dm_median_test(data; symmetric=true)
                end

                @test (result_nonsymm .< 0.05) == [false, true, true]
            end
        end

    end

end
