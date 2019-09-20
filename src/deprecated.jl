using Base: @deprecate

@deprecate joint_loglikelihood joint_gaussian_loglikelihood
@deprecate marginal_loglikelihood marginal_gaussian_loglikelihood

obs_arrangement(T::typeof(joint_loglikelihood)) = MatrixColsOfObs()
obs_arrangement(T::typeof(marginal_loglikelihood)) = MatrixColsOfObs()
