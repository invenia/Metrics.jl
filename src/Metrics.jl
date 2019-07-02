module Metrics

using Distributions
using LinearAlgebra: dot, norm

include("utils.jl")
include("regression.jl")
include("financial.jl")

export
    evaluate,
    squared_error,
    se,
    mean_squared_error,
    mse,
    root_mean_squared_error,
    rmse,
    normalised_root_mean_squared_error,
    nrmse,
    standardized_mean_squared_error,
    smse,
    absolute_error,
    ae,
    mean_absolute_error,
    mae,
    marginal_loglikelihood,
    joint_loglikelihood,
    picp,
    wpicp,
    apicp,
    expected_shortfall

end # module
