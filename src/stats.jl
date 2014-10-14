## stats.jl  Various ways of computing Baum Welch statistics for a GMM
## (c) 2013--2014 David A. van Leeuwen

mem=2.                          # Working memory, in Gig

function setmem(m::Float64) 
    global mem=m
end

import BigData.stats

## This function is admittedly hairy: in Octave this is much more
## efficient than a straightforward calculation.  I don't know if this
## holds for Julia.  We'd have to re-implement using loops and less
## memory.  I've done this now in several ways, it seems that the
## matrix implementation is always much faster.
 
## The shifting in dimensions (for Gaussian index k) is a nightmare.  

## stats(gmm, x) computes zero, first, and second order statistics of
## a feature file aligned to the gmm.  The statistics are ordered (ng
## * d), as by the general rule for dimension order in types.jl.
## Note: these are _uncentered_ statistics.

## you can dispatch this routine by specifying 3 parameters, 
## i.e., an unnamed explicit parameter order

function stats{T<:FloatingPoint}(gmm::GMM, x::Matrix{T}, order::Int)
    gmm.d == size(x,2) || error("dimension mismatch for data")
    if gmm.kind == :diag
        diagstats(gmm, x, order)
    elseif gmm.kind == :full
        fullstats(gmm, x, order)
    end
end

## For reasons of accumulation, this function returns a tuple
## (nx, loglh, N, F [S]) which should be easy to accumulate

## The memory footprint is sizeof(T) * ((4d +2) ng + (d + 4ng + 1) nx,
## This is not very efficient, since this is designed for speed, and
## wo don't want to do too much in-memory yet.  
##

function diagstats{T<:FloatingPoint}(gmm::GMM, x::Matrix{T}, order::Int)
    ng = gmm.n
    (nx, d) = size(x)
    prec = 1./gmm.Σ             # ng * d
    mp = gmm.μ .* prec              # mean*precision, ng * d
    ## note that we add exp(-sm2p/2) later to pxx for numerical stability
    a = gmm.w ./ (((2π)^(d/2)) * sqrt(prod(gmm.Σ,2))) # ng * 1
    
    sm2p = sum(mp .* gmm.μ, 2)      # sum over d mean^2*precision, ng * 1
    xx = x.^2                           # nx * d
    pxx = broadcast(+, sm2p', xx * prec') # nx * ng
    mpx = x * mp'                       # nx * ng
    L = broadcast(*, a', exp(mpx-0.5pxx)) # nx * ng, Likelihood per frame per Gaussian
    sm2p=pxx=mpx=0                   # save memory
    
    lpf=sum(L,2)                        # nx * 1, Likelihood per frame
    γ = broadcast(/, L, lpf .+ (lpf==0))' # ng * nx, posterior per frame per gaussian
    ## zeroth order
    N = reshape(sum(γ, 2), ng)          # ng * 1, vec()
    ## first order
    F =  γ * x                          # ng * d
    llh = sum(log(lpf))                 # total log likeliood
    if order==1
        return (nx, llh, N, F)
    else
        ## second order
        S = γ * xx                      # ng * d
        return (nx, llh, N, F, S)
    end
end

## this is a `slow' implementation, based on posterior()
function fullstats{T<:FloatingPoint}(gmm::GMM, x::Array{T,2}, order::Int)
    (nx, d) = size(x)
    ng = gmm.n
    γ, ll = posterior(gmm, x, true)
    llh = sum(logsumexp(broadcast(+, ll, log(gmm.w)'), 2))
    ## zeroth order
    N = vec(sum(γ, 1))
    ## first order
    F = γ' * x
    if order == 1
        return nx, llh, N, F
    end
    ## second order, this is harder now
    ## broadcast: ng * nx * d multiplications, hcat: nx * (ng*d)
    ## matmul: nx^2 * d^2 * ng multiplications...
    Sm = x' * hcat([broadcast(*, γ[:,j], x) for j=1:ng]...) # big bad matmul, nx
    S = Matrix{T}[Sm[:,(j-1)*d+1:j*d] for j=1:ng]
    ##S = [zeros(d,d) for i=1:ng]
    ##for i=1:nx
    ##    xi = x[i,:]
    ##    sxx = xi' * xi
    ##    for j=1:ng
    ##        S[j] += γ[i,j]*sxx
    ##    end
    return nx, llh, N, F, S
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
function stats{T<:FloatingPoint}(gmm::GMM, x::Matrix{T}; order::Int=2, parallel=false)
    ng = gmm.n
    (nx, d) = size(x)    
    bytes = sizeof(T) * ((4d +2)ng + (d + 4ng + 1)nx)
    blocks = iceil(bytes / (mem * (1<<30)))
    if parallel
        blocks= min(nx, max(blocks, nworkers()))
    end
    l = nx / blocks     # chop array into smaller pieces xx
    xx = Matrix{T}[x[round(i*l+1):round((i+1)l),:] for i=0:(blocks-1)]
    if parallel
        r = pmap(x->stats(gmm, x, order), xx)
    else
        r = map(x->stats(gmm, x, order), xx) # not very memory-efficient, but hey...
    end
    reduce(+, r)                # get +() from BigData.jl
end
## the reduce above needs the following
Base.zero{T}(x::Array{Matrix{T}}) = [zero(z) for z in x]
    
## This function calls stats() for the elements in d::Data, irrespective of the size, or type
function stats(gmm::GMM, d::Data; order::Int=2, parallel=false)
    if parallel
        r = dmap(x->stats(gmm, x, order=order, parallel=false), d)
        return reduce(+, r)
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
function csstats{T<:FloatingPoint}(gmm::GMM, x::Matrix{T}, order::Int=2)
    if order==1
        nx, llh, N, F = stats(gmm, x, order)
    else
        nx, llh, N, F, S = stats(gmm, x, order)
    end
    Nμ = broadcast(*, N, gmm.μ)
    f = (F - Nμ) ./ gmm.Σ
    if order==1
        return(N, f)
    else
        s = (S - (2F+Nμ).*gmm.μ) ./ gmm.Σ 
        return(N, f, s)
    end
end

## You can also get centered+scaled stats in a Cstats structure directly by 
## using the constructor with a GMM argument
CSstats{T<:FloatingPoint}(gmm::GMM, x::Matrix{T}) = CSstats(csstats(gmm, x, 1))

## centered stats, but not scaled by UBM covariance
function Stats{T<:FloatingPoint}(gmm::GMM, x::Matrix{T}, parallel=false) 
    nx, llh, N, F, S = stats(gmm, x, 2, parallel)
    Nμ = broadcast(*, N, gmm.μ)
    S -= (2F+Nμ) .* gmm.μ
    F -= Nμ
    Stats{T}(N, F, S)
end

