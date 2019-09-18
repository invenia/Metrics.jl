@testset "deprecated.jl" begin

    dist_scalar = Normal(0.1, 2)
    y_pred_scalar = [0.1, 0.2, 0.3]

    dist_mv = MvNormal(3, 1.5)
    y_pred_mv = [
        8.  10   9  11
        10   5   7  12
        10   7  10  1
    ]

    @testset "rename marginal_loglikelihood to marginal_gaussian_loglikelihood" begin
        @test isequal(
            marginal_loglikelihood(dist_scalar, y_pred_scalar),
            marginal_gaussian_loglikelihood(dist_scalar, y_pred_scalar),
        )

        @test isequal(
            marginal_loglikelihood(dist_mv, y_pred_mv),
            marginal_gaussian_loglikelihood(dist_mv, y_pred_mv),
        )
    end

    @testset "rename joint_loglikelihood to joint_gaussian_loglikelihood" begin
        @test isequal(
            joint_loglikelihood(dist_scalar, y_pred_scalar),
            joint_gaussian_loglikelihood(dist_scalar, y_pred_scalar),
        )

        @test isequal(
            joint_loglikelihood(dist_mv, y_pred_mv),
            joint_gaussian_loglikelihood(dist_mv, y_pred_mv),
        )
    end


end
