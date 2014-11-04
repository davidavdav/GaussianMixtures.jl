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
    α, β, m, ν, W = mstep(π, N, mx, S)
    hist = copy(g.hist)
    push!(hist, History("GMM converted to Variational GMM"))
    VGMM(g.n, g.d, π, α, β, m, ν, W, hist)
end
Base.copy(vg::VGMM) = VGMM(vg.n, vg.d, copy(vg.π), copy(vg.α), copy(vg.β),
                           copy(vg.m), copy(vg.ν), copy(vg.W), copy(vg.hist))

## sharpen VGMM to a GMM
## This currently breaks because my expected Λ are not positive definite
function GMM(vg::VGMM)
    w = vg.α / sum(vg.α)
    μ = vg.m
    Σ = similar(vg.W)
    for k=1:length(vg.W)
        Σ[k] = inv(vg.ν[k] * vg.W[k])
    end
    hist = copy(vg.hist)
    push!(hist, History("Variational GMM converted to GMM"))
    GMM(w, μ, Σ, hist, iround(sum(vg.α - vg.π.α0)))
end

## m-step given prior and stats
function mstep(π::GMMprior, N::Vector, mx::Matrix, S::Vector)
    ng = length(N)
    α = π.α0 + N                # ng, 10.58
    ν = π.ν0 + N + 1            # ng, 10.63
    β = π.β0 + N                # ng, 10.60
    m = similar(mx)             # ng * d
    W = similar(S)              # ng * (d*d)
    d = size(mx,2)
    limit = √ eps(eltype(N))
    for k=1:ng
        if N[k] > limit
            m[k,:] = (π.β0*π.m0' + N[k]*mx[k,:]) ./ β[k] # 10.61
            Δ = mx[k,:] - π.m0'
            ## do some effort to keep the matrix positive definite
            third = π.β0 * N[k] / (π.β0 + N[k]) * (Δ' * Δ) # guarantee symmety in Δ' Δ
            W[k] = inv(factorize(Hermitian(inv(π.W0) + N[k]*S[k] + third))) # 10.62
        else
            m[k,:] = zeros(d)
            W[k] = eye(d)
        end
    end
    return α, β, m, ν, W
end

## log(ρ_nk) from 10.46, start with a very slow implementation
## 10.46
function logρ(vg::VGMM, x::Matrix) 
    (nx, d) = size(x)
    d == vg.d || error("dimension mismatch")
    ng = vg.n
    Elogπ = digamma(vg.α) .- digamma(sum(vg.α)) # 10.66, size ng
    ElogdetΛ = similar(Elogπ) # size ng
    for k=1:ng
        ElogdetΛ[k] = sum(digamma(0.5(vg.ν[k] .+ 1 - [1:d]))) .+ d*log(2) .+ logdet(vg.W[k]) # 10.65
    end
    EμΛ = similar(x, nx, ng)    # nx * ng
    for i=1:nx
        for k=1:ng
            Δ = x[i,:] - vg.m[k,:]
            EμΛ[i,k] = d/vg.β[k] + vg.ν[k]* Δ*vg.W[k] ⋅ Δ
        end
    end
    broadcast(+, (Elogπ + 0.5ElogdetΛ .- 0.5d*log(2π))', -0.5EμΛ), Elogπ, ElogdetΛ
end

## 10.49
function rnk(vg::VGMM, x::Matrix)
#    ρ = exp(logρ(g, x))
#    broadcast(/, ρ, sum(ρ, 2))
    lρ, Elogπ, ElogdetΛ = logρ(vg, x)      # nx * ng
    broadcast!(-, lρ, lρ, logsumexp(lρ, 2))
    exp(lρ), Elogπ, ElogdetΛ
end

## We'd like to do this though stats(), but don't for now. 
## 10.51--10.53
function threestats(vg::VGMM, x::Matrix)
    ng = vg.n
    (nx, d) = size(x)
    r, Elogπ, ElogdetΛ = rnk(vg, x)
    r = r'                      # ng * nx, `wrong direction'
    N = vec(sum(r, 2))          # ng
    mx = broadcast(/, r * x, N) # ng * d
    S = similar(vg.W)           # ng * d*d
    for k = 1:ng
        S[k] = zeros(d,d)
        for i=1:nx
            Δ = x[i,:] - mx[k,:]
            S[k] += r[k,i]* Δ' * Δ
        end
        S[k] ./= N[k]
    end
    return N, mx, S, r, Elogπ, ElogdetΛ # r in transposed form...
end

## trace(A*B) = sum(A' .* B)
function trAB{T}(A::Matrix{T}, B::Matrix{T})
    nr, nc = size(A)
    size(B) == (nc, nr) || error("Inconsistent matrix size")
    s=zero(T)
    for i=1:nr for j=1:nc
        @inbounds s += A[i,j] * B[j,i]
    end end
    return s
end

## lower bound to the likelihood, using lots of intermediate results (10.2.2)
## ``We can straightforwardly evaluate the lower bound...''
function lowerbound(vg::VGMM, N::Vector, mx::Matrix, S::Vector,
                    r::Matrix, Elogπ::Vector, ElogdetΛ::Vector)
    ## shorthands that make the formulas easier to read...
    ng, d = vg.n, vg.d
    α0, β0, ν0, m0, W0  = vg.π.α0, vg.π.β0, vg.π.ν0, vg.π.m0, vg.π.W0 # prior vars
    W0inv = inv(W0)
    α, β, m, ν, W = vg.α, vg.β, vg.m, vg.ν, vg.W # VGMM vars
    gaussians = 1:ng
    ## B.79
    logB(W,ν) = -0.5ν*(logdet(W) + d*log(2)) - d*(d-1)/4*log(π) - sum(lgamma(0.5(ν+1-[1:d])))
    ## E[log p(x|Ζ,μ,Λ)] 10.71
    Elogll = 0.
    for k in gaussians
        Δ = mx[k,:] - m[k,:]   # 1 * d
        Elogll += 0.5N[k] * (ElogdetΛ[k] - d/β[k] - d*log(2π)
                             - ν[k] * (trAB(S[k], W[k]) + Δ * W[k] ⋅ Δ))
    end                        # 10.71
    ElogpZπ = sum(broadcast(*, r, Elogπ)) # E[log p(Z|π)] 10.72
    Elogpπ = lgamma(ng*α0) - ng*lgamma(α0) - (α0-1)sum(Elogπ) # E[log p(π)] 10.73
    ElogpμΛ = ng*logB(W0,ν0)   # E[log p(μ, Λ)] B.79
    for k in gaussians
        Δ = m[k,:] - m0'
        ElogpμΛ += 0.5(d*log(β0/(2π)) + (ν0-d)ElogdetΛ[k] - d*β0/β[k]
                       -β0*ν[k] * dot(Δ*W[k], Δ) - ν[k]*trAB(W0inv, W[k]))
    end                         # 10.74
    ## E[log q(Z)] 10.75
    ElogqZ = 0.
    for i = 1:size(r,2) for k in gaussians
        ElogqZ += r[k,i] * log(r[k,i])
    end end                     # 10.75
    Elogqπ = sum((α.-1).*Elogπ) + lgamma(sum(α)) - sum(lgamma(α)) # E[log q(π)] 10.76
    ## E[log q(μ,Λ)] 10.77
    ElogqμΛ = 0.
    for k in gaussians
        H = -logB(W[k],ν[k]) - 0.5(ν[k]-d-1)ElogdetΛ[k] + ν[k]*d/2 # H[q(Λ)] B.82
        ElogqμΛ += 0.5(ElogdetΛ[k] + d*log(β[k]/(2π)) - d) - H
    end                         # 10.77
    return Elogll + ElogpZπ + Elogpπ + ElogpμΛ + ElogqZ + Elogqπ + ElogqμΛ
end

## note: for matrix argument, Gaussian index must run down!
rmdisfunct(v::Vector, keep) = v[keep]
rmdisfunct(m::Matrix, keep) = m[keep,:]

## do exactly one update step for the VGMM, and return an estimate for the lower bound
## of the log marginal probability p(x)
function emstep!(vg::VGMM, x::Matrix)
    N, mx, S, r, Elogπ, ElogdetΛ = threestats(vg, x)
    ## remove defunct Gaussians
    keep = N .> √ eps(eltype(N))
    n = sum(keep)
    if n < vg.n
        N, mx, S, r, Elogπ, ElogdetΛ = map(x->rmdisfunct(x,keep), (N, mx, S, r, Elogπ, ElogdetΛ))
        vg.α, vg.β, vg.m, vg.ν, vg.W = map(x->rmdisfunct(x,keep), (vg.α, vg.β, vg.m, vg.ν, vg.W))
        vg.n = n
        addhist!(vg, @sprintf("dropping number of Gaussions to %d",n))
    end
    ## then compute the lowerbound
    L = lowerbound(vg, N, mx, S, r, Elogπ, ElogdetΛ)
    vg.α, vg.β, vg.m, vg.ν, vg.W = mstep(vg.π, N, mx, S)
    L
end

## This is called em!, but it is not really expectation maximization I think
function em!(vg::VGMM, x::Matrix; nIter=50)
    L = Float64[]
    for i=1:nIter
        push!(L, emstep!(vg, x))
        if i>1 && isapprox(L[i], L[i-1], rtol=0)
            nIter=i
            break
        end
        addhist!(vg, @sprintf("iteration %d, lowerbound %f", i, last(L)))
    end
    addhist!(vg, @sprintf("%d variational Bayes EM-like iterations, final lowerbound %f", nIter, last(L)))
    L
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
    Gaussian(μ, μ0, inv(β*Λ)) * Wishart(Λ, W, ν)
end 

