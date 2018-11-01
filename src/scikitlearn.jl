# This implements the ScikitLearn.jl interface and makes GMM compatible with
# ScikitLearn pipelines

import ScikitLearnBase

n_components(gmm::GMM) = size(gmm.μ, 1)

# We set d=1, but that's OK, we'll change it in fit!
GMM(; n_components=1, kind=:diag) = GMM(n_components, 2; kind=kind)
ScikitLearnBase.is_classifier(::GMM) = false
ScikitLearnBase.clone(gmm::GMM) =
    GMM(; n_components=n_components(gmm),
        kind=kind(gmm))

function ScikitLearnBase.get_params(gmm::GMM)
    return Dict(:n_components=>n_components(gmm))
end

function ScikitLearnBase.set_params!(gmm::GMM; params...)
    for (param, val) in params
        if param==:n_components
            # We don't have to update Σ, it'll be wiped in fit! anyway
            gmm.μ = zeros(val, 1)
        else
            throw(ArgumentError("Bad parameter: $param"))
        end
    end
    gmm
end

function Base.copy!(gmm_dest::GMM, gmm_src::GMM)
    # shallow copy - used below
    for f in fieldnames(typeof(gmm_dest))
        setfield!(gmm_dest, f, getfield(gmm_src, f))
    end
end

function ScikitLearnBase.fit!(gmm::GMM, X::AbstractMatrix, y=nothing)
    n = n_components(gmm)
    # Creating a temporary is not great, but it's negligible in the grand
    # scheme of thing.  We'd just need a slight refactor in
    # GaussianMixtures/src/train.jl to avoid it
    gmm_temp = GMM(n, X; kind=kind(gmm))
    copy!(gmm, gmm_temp)
    gmm
end

ScikitLearnBase.predict_log_proba(gmm::GMM, X) = log(gmmposterior(gmm, X)[1])
ScikitLearnBase.predict_proba(gmm::GMM, X) = gmmposterior(gmm, X)[1]
ScikitLearnBase.predict(gmm::GMM, X) =
    getindex.(argmax(ScikitLearnBase.predict_proba(gmm, X), dims=2), 2)

""" `density(gmm::GMM, X)` returns `log(P(X|μ, Σ))` """
function density(gmm::GMM, X)
    # Let mᵢ be "X came from mixture #i"
    # P(X|μ, Σ) = P(X|m₁, μ, Σ) * P(mᵢ) + P(X|m₂, μ, Σ) * P(m₂) + ...
    logPrior = reshape(log(gmm.w), 1, length(gmm.w))
    PX = logsumexp(broadcast(+, llpg(gmm, X), logPrior), 2)
    return squeeze(PX, 2)::Vector
end

# score_samples is underspecified by the scikit-learn API, so we're more or
# less free to return what we want
# http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html#sklearn.mixture.GMM
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
"""    score_samples(gmm::GMM, X::Matrix)
Return the per-sample likelihood of the data under the model. """
ScikitLearnBase.score_samples(gmm::GMM, X) = density(gmm, X)

# Used for cross-validation. Higher is better.
ScikitLearnBase.score(gmm::GMM, X) = avll(gmm, X)
