## distributions.jl (c) 2015 David A. van Leeuwen

## conversion to GMM
function GMM(m::MixtureModel{Multivariate,Continuous,MvNormal{PDMat,Vector{Float64}}})
    Σ = eltype(FullCov{Float64})[cholinv(c.Σ.mat) for c in components(m)]
    μ = hcat([c.μ for c in components(m)]...)'
    w = probs(m)
    n, d = size(μ)
    h = [History(@sprintf("Initialization from MixtureModel n=%d, d=%d, kind=full", n, d))]
    GMM(w, μ, Σ, h, 0)
end

function GMM(m::MixtureModel{Multivariate,Continuous,MvNormal{PDiagMat,Vector{Float64}}})
    Σ = hcat([c.Σ.diag for c in components(m)]...)'
    μ = hcat([c.μ for c in components(m)]...)'
    w = probs(m)
    n, d = size(μ)
    h = [History(@sprintf("Initialization from MixtureModel n=%d, d=%d, kind=diag", n, d))]
    GMM(w, μ, Σ, h, 0)
end

## conversion to MixtureModel
function Distributions.MixtureModel(gmm::GMM{Float64})
    if gmm.d == 1
        mixtures = [Normal(gmm.μ[i,1], gmm.Σ[i,1]) for i=1:gmm.n]
    elseif kind(gmm) == :full
        mixtures = [MvNormal(vec(gmm.μ[i,:]), covar(gmm.Σ[i])) for i=1:gmm.n]
    else
        mixtures = [MvNormal(vec(gmm.μ[i,:]), vec(gmm.Σ[i,:])) for i=1:gmm.n]
    end
    MixtureModel(mixtures, gmm.w)
end
