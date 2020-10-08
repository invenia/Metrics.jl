@testset "asymmetric.jl" begin
    @testset "asymmetric_absolute_error" begin
        y_true = 3
        y_pred_under = 1
        y_pred_over = 5

        @testset "exceptions" begin
            for p in (-1.1, 1.1)
                @test_throws DomainError asymmetric_absolute_error(
                    y_true, y_true; underpredict_penalty=p
                )
            end
        end

        @testset "default symmetric case" begin
            @test asymmetric_absolute_error(y_true, y_true) == 0
            @test evaluate(asymmetric_absolute_error, y_true, y_true) == 0

            err_over = asymmetric_absolute_error(y_true, y_pred_over)
            err_under = asymmetric_absolute_error(y_true, y_pred_under)
            @test err_under == err_over == 0.5 * 2
        end

        @testset "penalise under" begin
            penalty = 0.5
            @test asymmetric_absolute_error(
                y_true, y_true;
                underpredict_penalty=penalty
            ) == 0
            err_over = asymmetric_absolute_error(
                y_true, y_pred_over;
                underpredict_penalty=penalty
            )
            err_under = asymmetric_absolute_error(
                y_true, y_pred_under;
                underpredict_penalty=penalty
            )
            @test err_over < err_under
            @test err_over == 0.25 * 2
            @test err_under == 0.75 * 2
        end

        @testset "penalise over" begin
            penalty = -0.5
            @test asymmetric_absolute_error(
                y_true, y_true;
                underpredict_penalty=penalty
            ) == 0
            err_over = asymmetric_absolute_error(
                y_true, y_pred_over;
                underpredict_penalty=penalty
            )
            err_under = asymmetric_absolute_error(
                y_true, y_pred_under;
                underpredict_penalty=penalty
            )
            @test err_over > err_under
            @test err_over == 0.75 * 2
            @test err_under == 0.25 * 2
        end
    end
end
