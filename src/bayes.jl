## bayes.jl
## (c) 2014 David A. van Leeuwen
##
## Attempt to implement a Bayesian approach to EM for GMMs, along the lines of
## Christopher Bishop's book, section 10.2.

## This is only for practicing

## Wishart distribution {\cal W}(Λ, W, ν), but we write nu for ν. 
function Wishart(Λ::Matrix, W::Matrix, nu::Float64)
    ## check
    issym(W) || error("W must be symmetric")
    d = size(W,1)
    (d,d) == size(Λ) || error("Λ must be same size as W")
    ## norm
    B = det(W)^(-0.5nu) / (2^(0.5nu*d) * π^(0.25d*(d-1)))
    for i=1:d
        B /= gamma(0.5(nu+1-i))
    end
    invΛ = inv(Λ)
    ex = -0.5dot(inv(W),Λ)
    B * det(Λ)^(0.5(nu-d-1)) + exp(ex)
end

function Gaussian(x::Vector, μ::Vector, Σ::Matrix)
    d = length(μ)
    length(x) == d || error("Wrong dimension x")
    size(Σ) = (d,d) || error("Inconsistent size Σ")
    norm = inv((2π)^0.5d * sqrt(det(Σ)))
    ex = -0.5(x-μ)' * inv(Σ) * (x-μ)
    norm * exp(ex)
end

function GaussianWishart(μ::Vector, Λ::Matrix, μ0::Vector, β::Float64, W::Matrix, nu::Float64)
    Gaussian(μ, μ0, inv(βΛ)) * Wishart(Λ, W, nu)
end 

## log(ρ_nk) from 10.46
function logρ(x::Matrix, n::Int, k::Int, g::GMM, α::Vector, β::Float64, nu::Float64, W::Matrix)
    d = g.d
    Elogπ = digamma(α[k]) - digamma(mean(α)) # 10.66
    ElogdetΛ = sum(digamma(0.5(nu[k]+i-[1:d]))) + d*log(2) + logdet(W) # 10.65
    xx = x[n,:] - 
    EμkΛk = d/β + nu[k]*x[n,:]*
                                        
