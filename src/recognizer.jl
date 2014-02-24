## recognizer.jl.  Some routines for old-fashioned GMM-based (speaker) recognizers
## (c) 2013--2014 David A. van Leeuwen

## This function computes the `dotscoring' linear appoximation of a GMM/UBM log likelihood ratio
## of test data y using MAP adapted model for x.  
## We can compute this with just the stats:
function dotscore(x::Cstats, y::Cstats, r::Real=1.) 
    sum(broadcast(/, x.f, x.n + r) .* y.f)
end
## or directly from the UBM and the data x and y
dotscore{T<:Real}(gmm::GMM, x::Array{T,2}, y::Array{T,2}, r::Real=1.) =
    dotscore(Cstats(gmm, x), Cstats(gmm, y), r)

import Base.map

## Maximum A Posteriori adapt a gmm
function map{T<:Real}(gmm::GMM, x::Array{T,2}, r::Real=16.; means::Bool=true, weights::Bool=false, covars::Bool=false)
    nx, ll, N, F, S = stats(gmm, x)
    α = N ./ (N+r)
    g = GMM(gmm.n, gmm.d, gmm.kind)
    if weights
        g.w = α .* N / sum(N) + (1-α) .* gmm.w
        g.w ./= sum(g.w)
    else
        g.w = gmm.w
    end
    if means
        g.μ = broadcast(*, α./N, F) + broadcast(*, 1-α, gmm.μ)
    else
        g.μ = gmm.μ
    end
    if covars
        g.Σ = broadcast(*, α./N, S) + broadcast(*, 1-α, gmm.Σ .^2 + gmm.μ .^2) - g.μ .^2
    else
        g.Σ = gmm.Σ
    end
    addhist!(g,@sprintf "MAP adapted with %d data points relevance %3.1f %s %s %s" size(x,1) r means ? "means" : ""  weights ? "weights" : "" covars ? "covars" : "")
    return(g)
end

## compute a supervector from a MAP adapted utterance. 
function Base.vec(gmm::GMM, x::Matrix, r=16.)
    nx, ll, N, F, S = stats(gmm, x)
    α = N ./ (N+r)
    Δμ = broadcast(*, α./N, F) - broadcast(*, α, gmm.μ)
    v = Δμ .* sqrt(broadcast(/, gmm.w, gmm.Σ))
    return vec(v)
end