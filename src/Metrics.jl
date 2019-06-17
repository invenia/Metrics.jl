module Metrics

using Distributions

include("utils.jl")
include("regression.jl")

export
    evaluate,
    squared_error,
    marginal_loglikelihood,
    joint_loglikelihood,
    picp

end # module
