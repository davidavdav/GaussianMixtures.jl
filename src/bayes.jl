## bayes.jl
## (c) 2014, 2015 David A. van Leeuwen
##
## Attempt to implement a Bayesian approach to EM for GMMs, along the lines of
## Christopher Bishop's book, section 10.2.

## PLAN
## - convert stats collection to unnormalized stats and reduce results
## - optimize for speed

## Please note our pedantic use of the Greek symbols nu "ν" and alpha "α" (not to be confused with
## Latin v and a), and index "₀" and even superscript "⁻¹", which are both part of the identifiers
## e.g., W₀⁻¹ where others might write W0inv.

using SpecialFunctions: digamma
using SpecialFunctions: lgamma

## initialize a prior with minimal knowledge
function GMMprior(d::Int, alpha::T, beta::T) where {T<:AbstractFloat}
    m₀ = zeros(T, d)
    W₀ = eye(T, d)
    ν₀ = convert(T,d)
    GMMprior(alpha, beta, m₀, ν₀, W₀)
end
Base.copy(p::GMMprior) = GMMprior(p.α₀, p.β₀, copy(p.m₀), p.ν₀, copy(p.W₀))

## initialize from a GMM and nₓ, the number of points used to train the GMM.
function VGMM(g::GMM{T}, π::GMMprior{T}) where {T}
    nₓ = g.nx
    N = g.w * nₓ
    mx = g.μ
    if kind(g) == :diag
        S = covars(full(g))
    else
        S = covars(g)
    end
    α, β, m, ν, W = mstep(π, N, mx, S)
    hist = copy(g.hist)
    push!(hist, History("GMM converted to Variational GMM"))
    VGMM(g.n, g.d, π, α, β, m, ν, W, hist)
end
Base.copy(vg::VGMM) = VGMM(vg.n, vg.d, copy(vg.π), copy(vg.α), copy(vg.β),
                           copy(vg.m), copy(vg.ν), copy(vg.W), copy(vg.hist))

## W[k] really is chol(W_k, :U), use precision() to get it back
precision(c::AbstractTriangular) = c' * c
mylogdet(c::AbstractTriangular) = 2sum(log.(diag(c)))
mylogdet(m::Matrix) = logdet(m)

## sharpen VGMM to a GMM
## This currently breaks because my expected Λ are not positive definite
function GMM(vg::VGMM)
    w = vg.α / sum(vg.α)
    μ = vg.m
    Σ = similar(vg.W)
    for k=1:length(vg.W)
        Σ[k] = vg.W[k] * √vg.ν[k]
    end
    hist = copy(vg.hist)
    push!(hist, History("Variational GMM converted to GMM"))
    GMM(w, μ, Σ, hist, iround(sum(vg.α - vg.π.α₀)))
end

## m-step given prior and stats
function mstep(π::GMMprior, N::Vector{T}, mx::Matrix{T}, S::Vector) where {T}
    ng = length(N)
    α = π.α₀ .+ N               # ng, 10.58
    ν = π.ν₀ .+ N .+ 1          # ng, 10.63
    β = π.β₀ .+ N               # ng, 10.60
    m = similar(mx)             # ng × d
    W = Array{eltype(FullCov{T})}(undef, ng) # ng × (d*d)
    d = size(mx,2)
    limit = √ eps(eltype(N))
    W₀⁻¹ = inv(π.W₀)
    for k=1:ng
        if N[k] > limit
            m[k,:] = (π.β₀*π.m₀ + N[k]vec(mx[k,:])) ./ β[k] # 10.61 ## v0.5 arraymageddon
            Δ = vec(mx[k,:]) - π.m₀ ## v0.5 arraymageddon
            ## do some effort to keep the matrix positive definite
            third = π.β₀ * N[k] / (π.β₀ + N[k]) * (Δ * Δ') # guarantee symmety in Δ Δ'
            W[k] = cholesky(inv(cholesky(Symmetric(W₀⁻¹ + N[k]*S[k] +
                                                   third)))).U # 10.62
        else
            m[k,:] = zeros(d)
            W[k] = cholesky(eye(d))
        end
    end
    return α, β, m, ν, W
end

## this can be computed independently of the data
function expectations(vg::VGMM)
    ng, d = vg.n, vg.d
    Elogπ = digamma.(vg.α) .- digamma(sum(vg.α)) # 10.66, size ng
    ElogdetΛ = similar(Elogπ) # size ng
    for k in 1:ng
        ElogdetΛ[k] = sum(digamma.(0.5(vg.ν[k] .+ 1 .- collect(1:d)))) + d*log(2) + mylogdet(vg.W[k]) # 10.65
    end
    return Elogπ, ElogdetΛ
end

## log(ρ_nk) from 10.46, start with a very slow implementation, and optimize it further
## This is as the heart of VGMM training, it should be super efficient, possibly
## at the cost of readability.
## Removing the inner loop over nₓ, replacing it by a matmul led to a major speed increase
## not leading to further speed increase for (Δ * vg.W[k]) .* Δ
## - c = Δ * chol(vg.W[k],:L), c .* c
## - Base.BLAS.symm('R', 'U', vg.ν[k], vg.W[k], Δ) .* Δ
function logρ(vg::VGMM, x::Matrix, ex::Tuple)
    (nₓ, d) = size(x)
    d == vg.d || error("dimension mismatch")
    ng = vg.n
    EμΛ = similar(x, nₓ, ng)    # nₓ × ng
    Δ = similar(x)              # nₓ × d
    for k in 1:ng
        ### d/vg.β[k] + vg.ν[k] * (x_i - m_k)' W_k (x_i = m_k) forall i
        ## Δ = (x_i - m_k)' W_k (x_i = m_k)
        xμTΛxμ!(Δ, x, vec(vg.m[k,:]), vg.W[k])
        EμΛ[:,k] = d/vg.β[k] .+ vg.ν[k] * sum(abs2, Δ, dims=2)
    end
    Elogπ, ElogdetΛ = ex
    (Elogπ + 0.5ElogdetΛ .- 0.5d*log(2π))' .- 0.5EμΛ
end

## r_nk 10.49, plus E[log p(Z|π)] 10.72 and E[log q(Z)] 10.75
function rnk(vg::VGMM, x::Matrix, ex::Tuple)
#    ρ = exp(logρ(g, x))
#    broadcast(/, ρ, sum(ρ, 2))
    logr = logρ(vg, x, ex)        # nₓ × ng
    broadcast!(-, logr, logr, logsumexp(logr, 2))
    ## we slowly replace logr by r, this is just cosmetic!
    r = logr
    ## ElogpZπ = sum(broadcast(*, r, Elogπ)) # E[log p(Z|π)] 10.72
    ElogpZπ = 0.                # E[log p(Z|π)] 10.72
    ElogqZ = 0.                 # E[log q(Z)] 10.75
    for k in 1:vg.n
        elπ = ex[1][k]          # Elogπ[k]
        for i = 1:size(logr,1)
            rr = exp(logr[i,k])
            ElogpZπ += rr * elπ
            ElogqZ += rr * logr[i,k]
            r[i,k] = rr         # cosmetic, r is identical to logr
        end
    end
    r, ElogpZπ + ElogqZ
end

## OK, also do stats.
## Like for the GMM, we return nₓ, (some value), zeroth, first, second order stats
## All return values can be accumulated, except r, which we need for
## lowerbound ElogpZπ and ElogqZ
function stats(vg::VGMM, x::Matrix{T}, ex::Tuple) where {T}
    ng = vg.n
    (nₓ, d) = size(x)
    if nₓ == 0
        return 0, zero(RT), zeros(RT, ng), zeros(RT, ng, d), [zeros(RT, d,d) for k=1:ng]
    end
    r, ElogpZπqZ = rnk(vg, x, ex) # nₓ × ng
    N = vec(sum(r, dims=1))       # ng
    F = r' * x                    # ng × d
    ## S_k = sum_i r_ik x_i x_i'
    ## Sm = x' * hcat([broadcast(*, r[:,k], x) for k=1:ng]...)
    SS = similar(x, nₓ, d*ng)   # nₓ*nd*ng mem, nₓ*nd*ng multiplications
    for k in 1:ng
        for j=1:d for i=1:nₓ
            @inbounds SS[i,(k-1)*d+j] = r[i,k]*x[i,j]
        end end
    end
    Sm = x' * SS                # d * (d * ng) mem
    S = Matrix{T}[Sm[:,(k-1)*d+1:k*d] for k=1:ng]
    return nₓ, ElogpZπqZ, N, F, S
end

function stats(vg::VGMM, d::Data, ex::Tuple; parallel=false)
    if parallel
        r = dmap(x->stats(vg, x, ex), d)
        return reduce(+, r)
    else
        r = stats(vg, d[1], ex)
        for i=2:length(d)
            r += stats(vg, d[i], ex)
        end
        return r
    end
end

## trace(A*B) = sum(A' .* B)
function trAB(A::Matrix{T1}, B::Matrix{T2}) where {T1,T2}
    RT = promote_type(T1,T2)
    nr, nc = size(A)
    size(B) == (nc, nr) || error("Inconsistent matrix size")
    s=zero(RT)
    for i=1:nr for j=1:nc
        @inbounds s += A[i,j] * B[j,i]
    end end
    return s
end

## lower bound to the likelihood, using lots of intermediate results (10.2.2)
## ``We can straightforwardly evaluate the lower bound...''
function lowerbound(vg::VGMM, N::Vector, mx::Matrix, S::Vector,
                    Elogπ::Vector, ElogdetΛ::Vector, ElogpZπqZ)
    ## shorthands that make the formulas easier to read...
    ng, d = vg.n, vg.d
    α₀, β₀, ν₀, m₀, W₀  = vg.π.α₀, vg.π.β₀, vg.π.ν₀, vg.π.m₀, vg.π.W₀ # prior vars
    W₀⁻¹ = inv(W₀)
    α, β, m, ν, W = vg.α, vg.β, vg.m, vg.ν, vg.W # VGMM vars
    gaussians = 1:ng
    ## B.79
    logB(W, ν) = -0.5ν * (mylogdet(W) + d*log(2)) .- d*(d-1)/4*log(π) -
    sum(lgamma.(0.5(ν + 1 .- collect(1:d))))
    ## E[log p(x|Ζ,μ,Λ)] 10.71
    Elogll = 0.
    for k in gaussians
        Δ = vec(mx[k,:] - m[k,:])   # d ## v0.5 arraymageddon
        Wk = precision(W[k])
        Elogll += 0.5N[k] * (ElogdetΛ[k] - d/β[k] - d*log(2π)
                             - ν[k] * (trAB(S[k], Wk) + Wk * Δ ⋅ Δ)) # check chol efficiency
    end                        # 10.71
    ## E[log p(Z|π)] from rnk() 10.72
    Elogpπ = lgamma(ng*α₀) .- ng*lgamma(α₀) .- (α₀-1)sum(Elogπ) # E[log p(π)] 10.73
    ElogpμΛ = ng*logB(W₀, ν₀)   # E[log p(μ, Λ)] B.79
    for k in gaussians
        Δ = vec(m[k,:]) - m₀        # d ## v0.5 arraymageddon
        Wk = precision(W[k])
        ElogpμΛ += 0.5(d*log(β₀/(2π)) + (ν₀-d)ElogdetΛ[k] - d*β₀/β[k]
                       -β₀*ν[k] * Wk * Δ ⋅ Δ - ν[k]*trAB(W₀⁻¹, Wk))
    end                         # 10.74
    ## E[log q(Z)] from rnk() 10.75, combined with E[log p(Z|π)]
    Elogqπ = sum((α.-1).*Elogπ) + lgamma(sum(α)) - sum(lgamma.(α)) # E[log q(π)] 10.76
    ## E[log q(μ,Λ)] 10.77
    ElogqμΛ = 0.
    for k in gaussians
        H = -logB(W[k],ν[k]) - 0.5(ν[k]-d-1)ElogdetΛ[k] + ν[k]*d/2 # H[q(Λ)] B.82
        ElogqμΛ += 0.5(ElogdetΛ[k] + d*log(β[k]/(2π)) - d) - H
    end                         # 10.77
    return Elogll + ElogpZπqZ + Elogpπ + ElogpμΛ + Elogqπ + ElogqμΛ
end

## note: for matrix argument, Gaussian index must run down!
rmdisfunct(v::Vector, keep) = v[keep]
rmdisfunct(m::Matrix, keep) = m[keep, :]

## do exactly one update step for the VGMM, and return an estimate for the lower bound
## of the log marginal probability p(x)
function emstep!(vg::VGMM, x::DataOrMatrix)
    ## E-like step
    Elogπ, ElogdetΛ = expectations(vg)
    ## N, mx, S, r = threestats(vg, x, (Elogπ, ElogdetΛ))
    nₓ, ElogZπqZ, N, F, S = stats(vg, x, (Elogπ, ElogdetΛ))
    mx = F ./ N
    for k = 1:vg.n
        mk = vec(mx[k,:]) ## v0.5 arraymageddon
        S[k] = S[k] / N[k] - mk * mk'
    end
    ## remove defunct Gaussians
    keep = N .> √ eps(eltype(N))
    n = sum(keep)
    if n < vg.n
        N, mx, S, Elogπ, ElogdetΛ = map(x->rmdisfunct(x,keep), (N, mx, S, Elogπ, ElogdetΛ))
        vg.α, vg.β, vg.m, vg.ν, vg.W = map(x->rmdisfunct(x,keep), (vg.α, vg.β, vg.m, vg.ν, vg.W))
        vg.n = n
        addhist!(vg, @sprintf("dropping number of Gaussions to %d",n))
    end
    ## then compute the lowerbound
    L = lowerbound(vg, N, mx, S, Elogπ, ElogdetΛ, ElogZπqZ)
    ## and finally compute M-like step
    vg.α, vg.β, vg.m, vg.ν, vg.W = mstep(vg.π, N, mx, S)
    L, nₓ
end

## This is called em!, but it is not really expectation maximization I think
function em!(vg::VGMM, x; nIter=50)
    L = Float64[]
    nₓ = 0
    for i in 1:nIter
        lb, nₓ = emstep!(vg, x)
        push!(L, lb)
        if i > 1 && isapprox(L[i], L[i-1], rtol=0)
            nIter = i
            break
        end
        addhist!(vg, @sprintf("iteration %d, lowerbound %f", i, last(L)/nₓ/vg.d))
    end
    L ./= nₓ*vg.d
    addhist!(vg, @sprintf("%d variational Bayes EM-like iterations using %d data points, final lowerbound %f", nIter, nₓ, last(L)))
    L
end

## Not used

if false
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

function GaussianWishart(μ::Vector, Λ::Matrix, μ₀::Vector, β::Float64, W::Matrix, ν::Float64)
    Gaussian(μ, μ₀, inv(β*Λ)) * Wishart(Λ, W, ν)
end

end
