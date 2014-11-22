## recognizer.jl.  Some routines for old-fashioned GMM-based (speaker) recognizers
## (c) 2013--2014 David A. van Leeuwen

## This function computes the `dotscoring' linear appoximation of a GMM/UBM log likelihood ratio
## of test data y using MAP adapted model for x.  
## We can compute this with just the stats:
function dotscore(x::CSstats, y::CSstats, r::Real=1.) 
    sum(broadcast(/, x.f, x.n + r) .* y.f)
end
## or directly from the UBM and the data x and y
dotscore{T<:Real}(gmm::GMM, x::Matrix{T}, y::Matrix{T}, r::Real=1.) =
    dotscore(CSstats(gmm, x), CSstats(gmm, y), r)

import Base.map

## Maximum A Posteriori adapt a gmm
function map{T<:FloatingPoint}(gmm::GMM, x::Matrix{T}, r::Real=16.; means::Bool=true, weights::Bool=false, covars::Bool=false)
    nₓ, ll, N, F, S = stats(gmm, x)
    α = N ./ (N+r)
    if weights
        w = α .* N / sum(N) + (1-α) .* gmm.w
        w ./= sum(g.w)
    else
        w = gmm.w
    end
    if means
        μ = broadcast(*, α./N, F) + broadcast(*, 1-α, gmm.μ)
    else
        μ = gmm.μ
    end
    if covars
        kind(gmm) == :diag || error("Sorry, can't MAP adapt full covariance matrix GMM yet")
        Σ = broadcast(*, α./N, S) + broadcast(*, 1-α, gmm.Σ .^2 + gmm.μ .^2) - g.μ .^2
    else
        Σ = gmm.Σ
    end
    hist = vcat(gmm.hist, History(@sprintf "MAP adapted with %d data points relevance %3.1f %s %s %s" size(x,1) r means ? "means" : ""  weights ? "weights" : "" covars ? "covars" : ""))
    return GMM(w, μ, Σ, hist, nₓ) 
end

## compute a supervector from a MAP adapted utterance. 
function Base.vec(gmm::GMM, x::Matrix, r=16.)
    kind(gmm) == :diag || error("Sorry, can't compute MAP adapted supervector for full covariance matrix GMM yet")
    nₓ, ll, N, F, S = stats(gmm, x)
    α = N ./ (N+r)
    Δμ = broadcast(*, α./N, F) - broadcast(*, α, gmm.μ)
    v = Δμ .* sqrt(broadcast(/, gmm.w, gmm.Σ))
    return vec(v)
end
