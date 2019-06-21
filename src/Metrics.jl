module Metrics

using Distributions
using LinearAlgebra: dot

include("utils.jl")
include("regression.jl")

export
    evaluate,
    squared_error,
    marginal_loglikelihood,
    joint_loglikelihood,
    picp,
    wpicp,
    apicp

end # module
