module Metrics

using AxisKeys
using DataFrames: disallowmissing
using Distributions
using Distributions: GenericMvTDist
using Intervals: Interval
using KeyedDistributions
using LinearAlgebra: cholesky, det, dot, I, norm, tr
using NamedDims: unname
using ObservationDims
using PDMatsExtras
using SpecialFunctions: erf
using StatsBase
using StatsUtils
using StatsUtils: sqrtcov

include("piracy.jl")
include("utils.jl")
# regression
include("regression/asymmetric.jl")
include("regression/loglikelihood.jl")
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

export
    evaluate,
    # regression
    asymmetric_absolute_error,
    expected_squared_error, se,
    mean_squared_error, mse,
    mean_squared_error_to_mean, mse2m,
    root_mean_squared_error, rmse,
    root_mean_squared_error_to_mean, rmse2m,
    normalised_root_mean_squared_error, nrmse,
    standardized_mean_squared_error, smse,
    expected_absolute_error, ae,
    mean_absolute_error, mae,
    mean_absolute_scaled_error, mase,
    correct_sign_ratio,
    marginal_gaussian_loglikelihood,
    joint_gaussian_loglikelihood,
    loglikelihood,
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
    median_over_expected_shortfall, median_over_es, evano,
    mean_over_es,
    mean_minus_es,
    median_return,
    expected_shortfall, es, expected_windfall, ew,
    ew_over_es, expected_windfall_over_expected_shortfall,
    financial_summary,
    price_impact,
    # statistical
    bky_test,
    dm_mean_test,
    dm_median_test,
    subsample_ci,
    subsample_difference_ci,
    subsample_pct_difference_ci
end # module
