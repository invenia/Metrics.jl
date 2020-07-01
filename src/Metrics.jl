module Metrics

using AxisArrays
using Distributions
using IndexedDistributions
using Intervals
using LinearAlgebra: cholesky, det, dot, I, norm, tr
using NamedDims
using ObservationDims
using PSDMats
using SpecialFunctions
using StatsBase
using StatsUtils: sqrtcov

include("utils.jl")
# regression
include("regression/gaussian_loglikelihood.jl")
include("regression/picp.jl")
include("regression/potential_payoff.jl")
include("regression/simple.jl")
# financial
include("financial/expected_shortfall.jl")
include("financial/expected_windfall.jl")
include("financial/simple.jl")
include("financial/price_impact.jl")
# statistical
include("statistical/bky.jl")
include("statistical/diebold_mariano.jl")
include("statistical/kullback_leibler.jl")
include("statistical/subsample.jl")
# misc
include("evaluate.jl")
include("summary.jl")

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
    potential_payoff,
    prediction_interval_coverage_probability, picp,
    window_prediction_interval_coverage_probability, wpicp,
    adjusted_prediction_interval_coverage_probability, apicp,
    regression_summary,
    REGRESSION_METRICS,
    # model
    kullback_leibler, kl,
    # financials
    expected_return,
    volatility,
    sharpe_ratio,
    median_over_expected_shortfall, evano,
    mean_over_es,
    median_return,
    expected_shortfall, es, expected_windfall, ew,
    financial_summary,
    price_impact,
    # statistical
    bky_test,
    dm_mean_test,
    dm_median_test,
    subsample_ci
end # module
