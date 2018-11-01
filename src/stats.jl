## stats.jl  Various ways of computing Baum Welch statistics for a GMM
## (c) 2013--2014 David A. van Leeuwen

mem=0.1                          # Working memory for stats extraction, in Gig

## you can set the available working memory for stats calculation---this should be
## quite a bit less than the available memory, since there is (generally unknown) overhead
function setmem(gig::Float64)
    global mem=gig
end

## stats(gmm, x, order) computes zero, first, ... upto order (≤2) statistics of
## a feature file aligned to the gmm.  The statistics are ordered (ng
## * d), as by the general rule for dimension order in types.jl.
## Note: these are _uncentered_ statistics.

## you can dispatch this routine by specifying 3 parameters,
## i.e., an unnamed explicit parameter order

## For reasons of accumulation, this function returns a tuple
## (nx, loglh, N, F [S]).

## This function is admittedly hairy: in Octave this is much more
## efficient than a straightforward calculation.  I don't know if this
## holds for Julia.  We'd have to re-implement using loops and less
## memory.  I've done this now in several ways, it seems that the
## matrix implementation is always much faster.

## The shifting in dimensions (for Gaussian index k) is a nightmare.

## The memory footprint is sizeof(T) * ((2d + 2) ng + (d + ng + 1) nx, and
## results take an additional (2d +1) ng
## This is not very efficient, since this is designed for speed, and
## we don't want to do too much in-memory yet.
## Currently, I don't use a logsumexp implementation because of speed considerations,
## this might turn out numerically less stable for Float32

## diagonal covariance
function stats(gmm::GMM{GT,DCT}, x::Matrix{T}, order::Int) where DCT <: DiagCov{GT} where {GT, T}
    RT = promote_type(GT,T)
    (nₓ, d) = size(x)
    ng = gmm.n
    gmm.d == d || error("dimension mismatch for data")
    1 ≤ order ≤ 2 || error("order out of range")
    prec::Matrix{RT} = 1 ./ gmm.Σ             # ng × d
    mp::Matrix{RT} = gmm.μ .* prec          # mean*precision, ng × d
    ## note that we add exp(-sm2p/2) later to pxx for numerical stability
    a::Matrix{RT} = gmm.w ./ ((2π)^(d/2) * sqrt.(prod(gmm.Σ, dims=2))) # ng × 1
    sm2p::Matrix{RT} = dot(mp, gmm.μ, 2)    # sum over d mean^2*precision, ng × 1
    xx = x .* x                            # nₓ × d
##  γ = broadcast(*, a', exp(x * mp' .- 0.5xx * prec')) # nₓ × ng, Likelihood per frame per Gaussian
    γ = x * mp'                            # nₓ × ng, nₓ * d * ng multiplications
    LinearAlgebra.BLAS.gemm!('N', 'T', -one(RT)/2, xx, prec, one(RT), γ)
    for j = 1:ng
        la = log(a[j]) - 0.5sm2p[j]
        for i = 1:nₓ
            @inbounds γ[i,j] += la
        end
    end
    for i = 1:length(γ) @inbounds γ[i] = exp(γ[i]) end
    lpf=sum(γ,dims=2)                           # nₓ × 1, Likelihood per frame
    broadcast!(/, γ, γ, lpf .+ (lpf .== 0)) # nₓ × ng, posterior per frame per gaussian
    ## zeroth order
    N = vec(sum(γ, dims=1))          # ng, vec()
    ## first order
    F =  γ' * x                           # ng × d, Julia has efficient a' * b
    llh = sum(log.(lpf))                   # total log likeliood
    if order==1
        return (nₓ, llh, N, F)
    else
        ## second order
        S = γ' * xx                       # ng × d
        return (nₓ, llh, N, F, S)
    end
end

## Full covariance
## this is a `slow' implementation, based on posterior()
function stats(gmm::GMM{GT,FCT}, x::Array{T,2}, order::Int) where FCT <: FullCov{GT} where {GT, T}
    RT = promote_type(GT,T)
    (nₓ, d) = size(x)
    ng = gmm.n
    gmm.d == d || error("dimension mismatch for data")
    1 ≤ order ≤ 2 || error("order out of range")
    γ, ll = gmmposterior(gmm, x) # nₓ × ng, both
    llh = sum(logsumexp(ll .+ log.(gmm.w)', 2))
    ## zeroth order
    N = vec(sum(γ, dims=1))
    ## first order
    F = γ' * x
    if order == 1
        return nₓ, llh, N, F
    end
    ## S_k = Σ_i γ _ik x_i' * x
    S = Matrix{RT}[]
    γx = Array{RT}(undef, nₓ, d)
    @inbounds for k=1:ng
        #broadcast!(*, γx, γ[:,k], x) # nₓ × d mults
        for j = 1:d for i=1:nₓ
            γx[i,j] = γ[i,k]*x[i,j]
        end end
        push!(S, x' * γx)            # nₓ * d^2 mults
    end
    return nₓ, llh, N, F, S
end


## ## reduction function for the plain results of stats(::GMM)
## function accumulate(r::Vector{Tuple})
##     res = {r[1]...}           # first stats tuple, as array
##     for i=2:length(r)
##         for j = 1:length(r[i])
##             res[j] += r[i][j]
##         end
##     end
##     tuple(res...)
## end

## split computation up in parts, either because of memory limitations
## or because of parallelization
## You dispatch this by only using 2 parameters
function stats(gmm::GMM, x::Matrix{T}; order::Int=2, parallel=false) where T <: AbstractFloat
    parallel &= nworkers() > 1
    ng = gmm.n
    (nₓ, d) = size(x)
    if kind(gmm) == :diag
        bytes = sizeof(T) * ((2d +2)ng + (d + ng + 1)nₓ)
    elseif kind(gmm) == :full
        bytes = sizeof(T) * ((d + d^2 + 5nₓ + nₓ*d)ng + (2d + 2)nₓ)
    end
    blocks = ceil(Integer, bytes / (mem * (1<<30)))
    if parallel
        blocks= min(nₓ, max(blocks, nworkers()))
    end
    l = nₓ / blocks     # chop array into smaller pieces xx
    xx = Matrix{T}[x[round(Int, i*l+1):round(Int, (i+1)l),:] for i=0:(blocks-1)]
    if parallel
        r = pmap(x->stats(gmm, x, order), xx)
        reduce(+, r)                # get +() from BigData.jl
    else
        r = stats(gmm, popfirst!(xx), order)
        for x in xx
            r += stats(gmm, x, order)
        end
        r
    end
end
## the reduce above needs the following
Base.zero(x::Array{Matrix{T}}) where T = [zero(z) for z in x]

## This function calls stats() for the elements in d::Data, irrespective of the size, or type
function stats(gmm::GMM, d::Data; order::Int=2, parallel=false)
    if parallel
        return dmapreduce(x->stats(gmm, x, order=order, parallel=false), +, d)
    else
        r = stats(gmm, d[1], order=order, parallel=false)
        for i=2:length(d)
            r += stats(gmm, d[i], order=order, parallel=false)
        end
        return r
    end
end

## Same, but UBM centered+scaled stats
## f and s are ng * d
function csstats(gmm::GMM, x::DataOrMatrix{T}, order::Int=2) where T<:AbstractFloat
    kind(gmm) == :diag || error("Can only do centered and scaled stats for diag covariance")
    if order==1
        nₓ, llh, N, F = stats(gmm, x, order)
    else
        nₓ, llh, N, F, S = stats(gmm, x, order)
    end
    Nμ = N .* gmm.μ
    f = (F - Nμ) ./ gmm.Σ
    if order==1
        return(N, f)
    else
        s = (S + (Nμ-2F).*gmm.μ) ./ gmm.Σ
        return(N, f, s)
    end
end

## You can also get centered+scaled stats in a Cstats structure directly by
## using the constructor with a GMM argument
CSstats(gmm::GMM, x::DataOrMatrix) = CSstats(csstats(gmm, x, 1))

## centered stats, but not scaled by UBM covariance
## check full covariance...
function cstats(gmm::GMM, x::DataOrMatrix{T}, parallel=false) where T <: AbstractFloat
    nₓ, llh, N, F, S = stats(gmm, x, order=2, parallel=parallel)
    Nμ =  N .* gmm.μ
    ## center the statistics
    gmmkind = kind(gmm)
    if gmmkind == :diag
        S += (Nμ-2F) .* gmm.μ
    elseif gmmkind == :full
        for i in 1:length(S)
            μi = gmm.μ[i,:]
            Fμi = F[i,:]' * μi
            S[i] += N[i] * μi' * μi - Fμi' - Fμi
        end
    else
        error("Unknown kind")
    end
    F -= Nμ
    return N, F, S
end

Cstats(gmm::GMM, x::DataOrMatrix, parallel=false) = Cstats(cstats(gmm, x, parallel))

## conversion from Cstats to CSstats:
CSstats(gmm::GMM, cstats::Cstats) = CSstats(cstats.N, cstats.F ./ gmm.Σ)

## some convenience functions
Base.eltype(stats::Cstats{T}) where {T} = T
Base.size(stats::Cstats) = size(stats.F)
kind(stats::Cstats) = typeof(stats.S) <: Vector ? :full : :diag
Base.:+(a::Cstats, b::Cstats) = Cstats(a.N + b.N, a.F + b.F, a.S + b.S)
Base.zero(x::Cstats) = Cstats(zero(x.N), zero(x.F), zero(x.S))
