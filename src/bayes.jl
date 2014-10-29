## bayes.jl
## (c) 2014 David A. van Leeuwen
##
## Attempt to implement a Bayesian approach to EM for GMMs, along the lines of
## Christopher Bishop's book, section 10.2.

## This is only for practicing

require("bayestypes.jl")

## initialize a prior with minimal knowledge
function GMMprior{T<:FloatingPoint}(d::Int, alpha::T, beta::T)
    m0 = zeros(T, d)
    W0 = eye(T, d)
    nu0 = convert(T,d)
    GMMprior(alpha, beta, m0, W0, nu0)
end

## not sure this constructor is at all useful.
function VGMM(ng::Int, prior::GMMprior)
    d = length(prior.m0)
    alpha = prior.α0 * ones(ng)
    beta = prior.β0 * ones(ng)
    m = repmat(prior.m0', ng)
    W = [prior.W0 for k=1:ng]
    nu = prior.ν0 * ones(ng)
    hist = History(@sprintf("VGMM with %d Gaussians in %d dimensions initialized from prior", ng, d))
    VGMM(ng, d, :full, alpha, beta, m, nu, W, [hist])
end

## initialize from a GMM and data, seems we only need nx, perhaps store that in GMM?
function VGMM(g::GMM, x::Matrix, prior::GMMprior)
    (nx, d) = size(x)
    N = g.w * nx
    mx = g.μ
    if g.kind == :diag
        S = full(g).Σ
    else
        S = g.Σ
    end
    α, β, m, nu, W = mstep(prior, N, mx, S)
    hist = copy(g.hist)
    push!(hist, History("Initialized a Varitional GMM"))
    VGMM(g.n, d, :full, α, β, m, nu, W, hist)
end

## sharpen VGMM to a GMM
function GMM(v::VGMM)
    w = v.α / sum(v.α)
    μ = v.m
    Σ = similar(v.W)
    for k=1:length(v.W)
        Σ[k] = inv(v.nu[k] * v.W[k])
    end
    hist = copy(v.hist)
    push!(hist, History("Variational GMM converted to GMM"))
    GMM(:full, w, μ, Σ, hist)
end


## log(ρ_nk) from 10.46, start with a very slow implementation
## 10.46
function logρ(g::VGMM, x::Matrix) 
    (nx, d) = size(x)
    d == g.d || error("dimension mismatch")
    ng = g.n
    Elogπ = digamma(g.α) .- digamma(sum(g.α)) # 10.66, size ng 
    ElogdetΛ = similar(Elogπ) # size ng
    for k=1:ng
        ElogdetΛ[k] = sum(digamma(0.5(g.nu[k] .+ 1 .- [1:d]))) .+ d*log(2) .+ logdet(g.W[k]) # 10.65
    end
    EμkΛk = similar(x, nx, ng)
    for i=1:nx
        for k=1:ng
            xx = x[i,:] - g.m[k,:]
            EμkΛk[i,k] = d/g.β[k] + g.nu[k]* dot(xx*g.W[k], xx)
        end
    end
    broadcast(+, (Elogπ + 0.5ElogdetΛ .- 0.5d*log(2π))', -0.5EμkΛk)
end

## 10.49
function rnk(g::VGMM, x::Matrix) 
#    ρ = exp(logρ(g, x))
#    broadcast(/, ρ, sum(ρ, 2))
    lρ = logρ(g, x)
    broadcast!(-, lρ, lρ, logsumexp(lρ, 2))
    exp(lρ)
end

## We'd like to do this though stats(), but don't for now. 
## 10.51--10.53
function threestats(g::VGMM, x::Matrix)
    ng = g.n
    (nx, d) = size(x)
    r = rnk(g, x)'              # ng * nx
    N = vec(sum(r, 2))          # ng
    mx = broadcast(/, r * x, N) # ng * d
    S = similar(g.W)           # ng * d*d
    for k = 1:ng
        S[k] = zeros(d,d)
        for i=1:nx
            xx = x[i,:] - mx[k,:]
            S[k] += r[k,i]* xx' * xx
        end
        S[k] /= N[k]
    end
    return N, mx, S
end

## m-step given prior and stats
function mstep(prior::GMMprior, N, mx, S)
    α = prior.α0 + N           # ng, 10.58
    nu = prior.ν0 + N + 1      # ng, 10.63
    β = prior.β0 + N           # ng, 10.60
    m = similar(mx)            # ng * d
    W = similar(S)             # ng * (d*d)
    for k=1:g.n
        m[k,:] = (prior.β0*prior.m0' + N[k]*mx[k,:]) ./ β[k] # 10.61
        xx = mx[k,:] - prior.m0'
        third = prior.β0 * N[k] / (prior.β0 + N[k]) * xx' * xx
        W[k] = inv(inv(prior.W0) + N[k]*S[k] + third) # 10.62
    end
    return α, β, m, nu, W
end

## do exactly one update step for the VGMM
function emstep!(g::VGMM, x::Matrix, prior::GMMprior)
    N, mx, S = threestats(g, x)
    g.α, g.β, g.m, g.nu, g.W = mstep(prior, N, mx, S)
    g
end

function em!(g::VGMM, x::Matrix, prior::GMMprior; nIter=50)
    for i=1:nIter
        emstep!(g, x, prior)
    end
end

## Not used

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
#    invΛ = inv(Λ)
    ex = -0.5dot(inv(W),Λ)
    B * det(Λ)^(0.5(nu-d-1)) * exp(ex)
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

