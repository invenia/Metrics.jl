module Metrics

using Distributions
using LinearAlgebra: dot, norm
using NamedDims
using StatsUtils: sqrtcov

include("evaluate.jl")
include("utils.jl")
include("regression.jl")
include("regression_picp.jl")
include("financial.jl")
include("price_impact.jl")

export
    evaluate,
    # regression
    squared_error, se,
    mean_squared_error, mse,
    root_mean_squared_error, rmse,
    normalised_root_mean_squared_error, nrmse,
    standardized_mean_squared_error, smse,
    absolute_error, ae,
    mean_absolute_error, mae,
    marginal_loglikelihood,
    joint_loglikelihood,
    picp,
    wpicp,
    apicp,
    # financials
    expected_return,
    volatility,
    sharpe_ratio,
    expected_shortfall, es,
    sample_expected_shortfall, ses,
    analytic_expected_shortfall, aes
end # module
