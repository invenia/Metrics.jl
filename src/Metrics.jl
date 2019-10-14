module Metrics

using AxisArrays
using Distributions
using IndexedDistributions
using LinearAlgebra: cholesky, det, dot, I, norm, tr
using NamedDims
using PSDMats
using SpecialFunctions
using StatsUtils: sqrtcov

include("evaluate.jl")
include("utils.jl")
include("regression.jl")
include("regression_picp.jl")
include("kullback_leibler.jl")
include("financial.jl")
include("price_impact.jl")
include("summaries.jl")
include("deprecated.jl")



export
    evaluate,
    # regression
    expected_squared_error, se,
    mean_squared_error, mse,
    root_mean_squared_error, rmse,
    normalised_root_mean_squared_error, nrmse,
    standardized_mean_squared_error, smse,
    expected_absolute_error, ae,
    mean_absolute_error, mae,
    mean_absolute_scaled_error, mase,
    marginal_gaussian_loglikelihood,
    joint_gaussian_loglikelihood,
    prediction_interval_coverage_probability, picp,
    window_prediction_interval_coverage_probability, wpicp,
    adjusted_prediction_interval_coverage_probability, apicp,
    regression_summary,
    REGRESSION_METRICS,
    # divergence
    kullback_leibler, kl,
    # financials
    expected_return,
    volatility,
    sharpe_ratio,
    median_over_expected_shortfall, evano,
    median_return,
    expected_shortfall, es,
    financial_summary
end # module
