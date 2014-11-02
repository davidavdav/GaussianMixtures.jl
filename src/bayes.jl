## bayes.jl
## (c) 2014 David A. van Leeuwen
##
## Attempt to implement a Bayesian approach to EM for GMMs, along the lines of
## Christopher Bishop's book, section 10.2.

## This is only for practicing

## initialize a prior with minimal knowledge
function GMMprior{T<:FloatingPoint}(d::Int, alpha::T, beta::T)
    m0 = zeros(T, d)
    W0 = eye(T, d)
    ν0 = convert(T,d)
    GMMprior(alpha, beta, m0, W0, ν0)
end
Base.copy(p::GMMprior) = GMMprior(p.α0, p.β0, copy(p.m0), copy(p.W0), p.ν0)

## initialize from a GMM and nx, the number of points used to train the GMM.
function VGMM(g::GMM, π::GMMprior)
    nx = g.nx
    N = g.w * nx
    mx = g.μ
    if kind(g) == :diag
        S = full(g).Σ
    else
        S = g.Σ
    end
    α, β, m, ν, W, keep = mstep(π, N, mx, S)
    hist = copy(g.hist)
    push!(hist, History("GMM converted to Varitional GMM"))
    VGMM(g.n, g.d, π, α, β, m, ν, W, hist)
end

## sharpen VGMM to a GMM
## This currently breaks because my expected Λ are not positive definite
function GMM(v::VGMM)
    w = v.α / sum(v.α)
    μ = v.m
    Σ = similar(v.W)
    for k=1:length(v.W)
        Σ[k] = inv(v.ν[k] * v.W[k])
    end
    hist = copy(v.hist)
    push!(hist, History("Variational GMM converted to GMM"))
    GMM(w, μ, Σ, hist, iround(sum(v.α)))
end

## m-step given prior and stats
function mstep(π::GMMprior, N, mx, S)
    ng = length(N)
    α = π.α0 + N           # ng, 10.58
    ν = π.ν0 + N + 1      # ng, 10.63
    β = π.β0 + N           # ng, 10.60
    m = similar(mx)            # ng * d
    W = similar(S)             # ng * (d*d)
    d = size(mx,2)
    limit = sqrt(eps(eltype(N)))
    keep = trues(length(N))
    for k=1:ng
        if N[k] > limit
            m[k,:] = (π.β0*π.m0' + N[k]*mx[k,:]) ./ β[k] # 10.61
            Δ = mx[k,:] - π.m0'
            third = π.β0 * N[k] / (π.β0 + N[k]) * Δ' * Δ
            W[k] = inv(inv(π.W0) + N[k]*S[k] + third) # 10.62
        else
            keep[k] = false
            m[k,:] = zeros(d)
            W[k] = eye(d)
        end
    end
    return α, β, m, ν, W, keep
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
        ElogdetΛ[k] = sum(digamma(0.5(g.ν[k] .+ 1 .- [1:d]))) .+ d*log(2) .+ logdet(g.W[k]) # 10.65
    end
    EμΛ = similar(x, nx, ng)
    for i=1:nx
        for k=1:ng
            xx = x[i,:] - g.m[k,:]
            EμΛ[i,k] = d/g.β[k] + g.ν[k]* dot(xx*g.W[k], xx)
        end
    end
    broadcast(+, (Elogπ + 0.5ElogdetΛ .- 0.5d*log(2π))', -0.5EμΛ)
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
    r = rnk(g, x)'              # ng * nx, `wrong direction'
    N = vec(sum(r, 2))          # ng
    mx = broadcast(/, r * x, N) # ng * d
    S = similar(g.W)            # ng * d*d
    for k = 1:ng
        S[k] = zeros(d,d)
        for i=1:nx
            xx = x[i,:] - mx[k,:]
            S[k] += r[k,i]* xx' * xx
        end
        S[k] ./= N[k]
    end
    return N, mx, S
end

## do exactly one update step for the VGMM
function emstep!(g::VGMM, x::Matrix)
    N, mx, S = threestats(g, x)
    g.α, g.β, g.m, g.ν, g.W, keep = mstep(g.π, N, mx, S)
    n = sum(keep)
    if n<g.n
        ## only keep useful Gaussians...
        for f in [:α, :β, :ν, :W]
            setfield!(g, f, getfield(g, f)[keep])
        end
        g.m = g.m[keep,:]
        g.n = n
        addhist!(g, string("dropping number of Gaussions to ",n))
    end
    g
end

function em!(g::VGMM, x::Matrix; nIter=50)
    for i=1:nIter
        emstep!(g, x)
    end
    addhist!(g, string(nIter, " variational Bayes EM-like iterations"))
end

## Not used

## Wishart distribution {\cal W}(Λ, W, ν).
function Wishart(Λ::Matrix, W::Matrix, ν::Float64)
    ## check
    issym(W) || error("W must be symmetric")
    d = size(W,1)
    (d,d) == size(Λ) || error("Λ must be same size as W")
    ## norm
    B = det(W)^(-0.5ν) / (2^(0.5ν*d) * π^(0.25d*(d-1)))
    for i=1:d
        B /= gamma(0.5(ν+1-i))
    end
#    invΛ = inv(Λ)
    ex = -0.5dot(inv(W),Λ)
    B * det(Λ)^(0.5(ν-d-1)) * exp(ex)
end

function Gaussian(x::Vector, μ::Vector, Σ::Matrix)
    d = length(μ)
    length(x) == d || error("Wrong dimension x")
    size(Σ) = (d,d) || error("Inconsistent size Σ")
    norm = inv((2π)^0.5d * sqrt(det(Σ)))
    ex = -0.5(x-μ)' * inv(Σ) * (x-μ)
    norm * exp(ex)
end

function GaussianWishart(μ::Vector, Λ::Matrix, μ0::Vector, β::Float64, W::Matrix, ν::Float64)
    Gaussian(μ, μ0, inv(βΛ)) * Wishart(Λ, W, ν)
end 

