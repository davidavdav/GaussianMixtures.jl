## GMM.jl  Useful functions for a Gaussia Mixture Model
## (c) 2013 David A. van Leeuwen

## This module also contains some rudimentary code for speaker
## recognition, perhaps this should move to another module.

module GMMs

## some init code.  Turn off subnormal computation, as it is slow.  This is a global setting...
ccall(:jl_zero_subnormals, Bool, (Bool,), true)

include("gmmtypes.jl")

export GMM, Cstats, History, split, em!, map, llpg, post, history, show, stats, cstats, dotscore, savemat, readmat, nparams

nparams(GMM) = sum(map(length, (GMM.w, GMM.μ, GMM.Σ)))

using Misc
using Clustering

import Base.copy

function addhist!(gmm::GMM, s::String) 
    gmm.hist = vcat(gmm.hist, History(s))
end

## copy a GMM, deep-copying all internal information. 
function copy(gmm::GMM)
    x = GMM(gmm.n, gmm.d, gmm.kind)
    x.w = copy(gmm.w)
    x.μ = copy(gmm.μ)
    x.Σ = copy(gmm.Σ)
    addhist!(x,"copy")
    x
end

## Greate a GMM with only one mixture and initialize it to ML parameters
function GMM{T<:Real}(x::Array{T,2})
    d=size(x, 2)
    gmm = GMM(1,d)
    gmm.μ = mean(x, 1)
    gmm.Σ = var(x, 1)
    addhist!(gmm, @sprintf "Initlialized single Gaussian with %d data points" size(x,1))
    gmm
end

function GMM{T<:Real}(n::Int, x::Array{T,2}, method::Symbol=:kmeans; nInit::Int=50, nIter::Int=10, nFinal::Int=nIter, fast=true, logll=true)
    if method==:split
        GMM2(n, x, nIter=nIter, nFinal=nFinal, fast=fast, logll=logll)
    elseif method==:kmeans
        GMMk(n, x, nInit=nInit, nIter=nIter)
    else
        error("Unknown method ", method)
    end
end

## initialize GMM using Clustering.kmeans (which uses a method similar to kmeans++)
function GMMk{T<:Real}(n::Int, x::Array{T,2}; nInit::Int=50, nIter::Int=10, logll=true)
    gmm = GMM(n, ncol(x))
    km = kmeans(convert(Array{Float64},x'), n, max_iter=nInit, display = logll ? :iter : :none)
    gmm.μ = km.centers'
    gmm.Σ = convert(Array{Float64,2},vcat(map(i -> var(x[km.assignments.==i,:],1), 1:n)...))
    gmm.w = km.counts ./ sum(km.counts)
    addhist!(gmm, string("K-means with ", nrow(x), " data points using ", km.iterations, " iteration"))
    em!(gmm, x; nIter=nIter, logll=logll)
    gmm
end    

## Train a GMM by consecutively splitting all means.  n most be a power of 2
## This kind of initialization is deterministic, but doesn't work particularily well, its seems
## We start with one Gaussian, and consecutively split.  
function GMM2{T<:Real}(n::Int, x::Array{T,2};nIter::Int=10, nFinal::Int=nIter, fast=true, logll=true)
    log2n = int(log2(n))
    @assert 2^log2n == n
    gmm=GMM(x)
    tll = avll(gmm,x)
    println("0: avll = ", tll)
    for i=1:log2n
        gmm=split(gmm)
        avll = em!(gmm, x; logll=true, nIter=i==log2n ? nFinal : nIter, fast=fast, logll=logll)
        println(i, ": avll = ", avll)
        tll = vcat(tll, avll)
    end
    println(tll)
    gmm
end
GMM{T<:Real}(n::Int,x::Vector{T};nIter::Int=10) = GMM(n, reshape(x, length(x), 1);  nIter=nIter)

## Average log-likelihood per data point and per dimension for a given GMM 
function avll{T<:Real}(gmm::GMM, x::Array{T,2})
    @assert gmm.d == size(x,2)
    llpfpg = llpg(gmm, x)
    llpf = log(exp(llpfpg) * gmm.w)
    mean(llpf) / gmm.d
end
    

import Base.split
## Split a gmm in order to to double the amount of gaussians
function split(gmm::GMM; minweight::Real=1e-5, covfactor::Real=0.2)
    ## In this function i, j, and k all index Gaussians
    maxi = reverse(sortperm(gmm.w))
    offInd = find(gmm.w .< minweight)
    if (length(offInd)>0) 
        println("Removing Gaussians with no data");
    end
    for i=1:length(offInd) 
        gmm.w[maxi[i]] = gmm.w[offInd[i]] = gmm.w[maxi[i]]/2;
        gmm.μ[offInd[i],:] = gmm.μ[maxi[i],:] + covfactor*sqrt((gmm.Σ[maxi[i],:]))
        gmm.μ[maxi[i],:] = gmm.μ[maxi[i],:] - covfactor*sqrt((gmm.Σ[maxi[i],:]))
    end
    new = GMM(2gmm.n, gmm.d, gmm.kind)
    for i=1:gmm.n
        j = 2i-1 : 2i
        new.w[j] = gmm.w[i]/2
        for k=j 
            new.μ[k,:] = gmm.μ[i,:] + sign(k-2i+0.5) * covfactor * sqrt(gmm.Σ[i,:])
            new.Σ[k,:] = gmm.Σ[i,:]
        end
    end
    new.hist = vcat(gmm.hist, History(@sprintf "split to %d Gaussians" new.n))
    new
end

# This function runs the Expectation Maximization algorithm on the GMM, and returns
# the log-likelihood history, per data frame per dimension
function em!{T<:Real}(gmm::GMM, x::Array{T,2}; nIter::Int = 10, varfloor::Real=1e-3, logll=true, fast=true) 
    @assert size(x,2)==gmm.d
    MEM = 2*(2<<30)             # 2GB
    d = gmm.d                   # dim
    ng = gmm.n                  # n gaussians
    initc = gmm.Σ
    blocksize = floor(MEM/((3+3ng)sizeof(Float64)))
    nf = size(x, 1)             # n frames
    ll = zeros(nIter)
    nx = 0
    for i=1:nIter
        ## E-step
        b = 0                  # pointer to start
        denom = zeros(ng)
        sx = sxx = zeros(ng,d)
        while (b < nf) 
            e=min(b+blocksize, nf)
            xx = x[b+1:e,:]
            nxx = e-b
            if fast
                (N, F, S, llpf) = stats(gmm, xx, 2, llpf=true)
                denom += N
                sx += F
                sxx += S
                if (logll || i==nIter) 
                    ll[i] += sum(log(llpf))
                end
            else
                (p,a) = post(gmm, xx) # nx * ng
                denom += sum(p,1)'
                sx += p' * xx
                sxx += p' * xx.^2
                if (logll || i==nIter) 
                    ll[i] += sum(log(a*gmm.w))
                end
            end
            b += nxx             # b=e
        end
        nx = b
        ## M-step
        gmm.w = denom[:]/nx
        gmm.μ = broadcast(/, sx, denom)
        gmm.Σ = broadcast(/, sxx, denom) - gmm.μ.^2
        ## var flooring
        tooSmall = any(gmm.Σ .< varfloor, 2)
        if (any(tooSmall))
            println("Variances had to be floored")
            ind = find(tooSmall)
            gmm.Σ[ind,:] = initc[ind,:]
        end
    end
    ll /= nx * d
    addhist!(gmm,@sprintf "EM with %d data points %d iterations avll %f" nx nIter ll[nIter])
    ll
end
    
## this function returns the contributions of the individual Gaussians to the LL
## ll_ij = log p(x_i | gauss_j)
function llpg{T<:Real}(gmm::GMM, x::Array{T,2})
    (nx, d) = size(x)
    ng = gmm.n
    @assert d==gmm.d    
    if (gmm.kind==:diag)
        ll = zeros(nx, ng)
        normalization = log((2π)^(d/2) * sqrt(prod(gmm.Σ,2))) # row 1...ng
        for j=1:ng
            Δ = broadcast(-, x, gmm.μ[j,:]) # nx * d
            ll[:,j] = -0.5sum(broadcast(/, Δ.^2, gmm.Σ[j,:]),2) - normalization[j]
        end
    else
        error("Unimplemented kind")
    end
    ll
end

## this function returns the posterior for component j: p_ij = p(j | gmm, x_i)
function post{T}(gmm, x::Array{T,2})      # nx * ng
    (nx, d) = size(x)
    ng = gmm.n
    @assert d==gmm.d
    a = exp(llpg(gmm, x))
    p = broadcast(*, a, gmm.w')
    sp = sum(p, 2)
    sp += sp==0       # prevent possible /0
    p = broadcast(/, p, sp)
    (p, a)
end

function history(gmm::GMM) 
    t0 = gmm.hist[1].t
    for h=gmm.hist
        print(@sprintf "%6.3f\t%s\n" h.t-t0 h.s)
    end
end

import Base.show

## we could improve this a lot
function show(io::IO, gmm::GMM) 
    print(io, @sprintf "GMM with %d components in %d dimensions and %s covariance\n" gmm.n gmm.d gmm.kind)
    for j=1:gmm.n
        print(io, @sprintf "Mix %d: weight %f, mean:\n" j gmm.w[j]);
        print(io, gmm.μ[j,:])
        print(io, "covariance:\n")
        print(io, gmm.Σ[j,:])
    end
end
    
## This function is admittedly hairy: in Octave this is much more efficient than a 
## straightforward calculation.  I don't know if this holds for Julia.  We'd have to re-implement 
## using loops and less memory.  I've done this now in several ways, it seems that the matrix 
## implementation is always much faster. 
 
## The shifting in dimensions (for Gaussian index k) is a nightmare.  

## stats(gmm, x) computes zero, first, and second order statistics of a feature 
## file aligned to the gmm.  The statistics are ordered (ng * d), as by the general 
## rule for dimension order in types.jl.  Note: these are _uncentered_ statistics. 
function stats{T<:Real}(gmm::GMM, x::Array{T,2}, order::Int=2; llpf=false)
    ng = gmm.n
    (nx, d) = size(x)
    @assert d==gmm.d
    prec = 1./gmm.Σ             # ng * d
    mp = gmm.μ .* prec              # mean*precision, ng * d
    sm2p = sum(mp .* gmm.μ, 2)      # sum over d mean^2*precision, ng * 1
    a = gmm.w ./ (((2π)^(d/2)) * sqrt(prod(gmm.Σ,2))) .* exp(-sm2p/2) # ng * 1
    sm2p=0                      # save memory
    
    xx = x.^2                           # nx * d
    pxx = xx * prec'                    # nx * ng
    mpx = x * mp'                       # nx * ng
    L = broadcast(*, a', exp(mpx-pxx/2)) # nx * ng, Likelihood per frame per Gaussian
    pxx=mpx=0                   # save memory
    
    denom=sum(L,2)                        # nx * 1, Likelihood per frame
    γ = broadcast(/, L, denom + (denom==0))' # ng * nx, posterior per frame per gaussian
    ## zeroth order
    N = reshape(sum(γ, 2), ng)               # ng * 1
    ## first order
    F =  γ * x                  # ng * d
    if order==1
        if llpf
            return (N, F, denom)
        else
            return(N, F)
        end
    else
        ## second order
        S = γ * xx                  # ng * d
        if llpf
            return (N, F, S, denom)
        else
            return (N, F, S)
        end
    end
end

## Same, but UBM centered stats
function cstats{T<:Real}(gmm::GMM, x::Array{T,2}, order::Int=2)
    if order==1
        (N,F) = stats(gmm, x, order)
    else
        (N, F, S) = stats(gmm, x)
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
## You can also get centered stats in a Cstats structure directly by 
## using the constructor with a GMM argument
Cstats{T<:Real}(gmm::GMM, x::Array{T,2}) = Cstats(cstats(gmm, x, 1))

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
    (n, F, S) = stats(gmm, x)
    α = n ./ (n+r)
    g = GMM(gmm.n, gmm.d, gmm.kind)
    if weights
        g.w = α .* n / sum(n) + (1-α) .* gmm.w
        g.w ./= sum(g.w)
    else
        g.w = gmm.w
    end
    if means
        g.μ = broadcast(*, α./n, F) + broadcast(*, 1-α, gmm.μ)
    else
        g.μ = gmm.μ
    end
    if covars
        g.Σ = broadcast(*, α./n, S) + broadcast(*, 1-α, gmm.Σ .^2 + gmm.μ .^2) - g.μ .^2
    else
        g.Σ = gmm.Σ
    end
    addhist!(g,@sprintf "MAP adapted with %d data points relevance %3.1f %s %s %s" nrow(x) r means ? "means" : ""  weights ? "weights" : "" covars ? "covars" : "")
    return(g)
end

## This code is for exchange with our octave / matlab based system

using MAT

## for compatibility with good-old Netlab's GMM
function savemat(file::String, gmm::GMM) 
    addhist!(gmm,string("GMM written to file ", file))
    matwrite(file, 
             { "gmm" =>         # the default name
              { "ncentres" => gmm.n,
               "nin" => gmm.d,
               "covar_type" => string(gmm.kind),
               "priors" => gmm.w,
               "centres" => gmm.μ,
               "covars" => gmm.Σ,
               "history_s" => string([h.s for h=gmm.hist]),
               "history_t" => [h.t for h=gmm.hist]
               }})
end
                                                                                    
function readmat{T}(file, ::Type{T})
    vars = matread(file)
    if T==GMM
        g = vars["gmm"]        
        n = int(g["ncentres"])
        d = int(g["nin"])
        kind = g["covar_type"]
        gmm = GMM(n, d, :diag)  # I should parse this
        gmm.w = reshape(g["priors"], n)
        gmm.μ = g["centres"]
        gmm.Σ = g["covars"]
        hist_s = split(get(g, "history_s", "No original history"), "\n")
        hist_t = get(g, "history_t", time())
        gmm.hist =  vcat([History(t,s) for (t,s) = zip(hist_t, hist_s)], 
                         History(string("GMM read from file ", file)))
    else
        error("Unknown type")
    end
    gmm
end

using Distributions

## this could be better
function test_GMM()
    N = 4
    d = 2
    nx = 100
    φ = π/2
    data = zeros(nx * N, d)
    for j=1:N
        data[(j-1)*nx+1 : j*nx, : ] = rand(DiagNormal(5*[sin(φ + 2π*j/N), cos(φ + 2π*j/N)], ones(d)), nx)'
    end
    data
#    g = GMM(N, d)
#    g.μ = randn(N, d)
#    for i=1:10
#        println(em!(g, data; nIter=5, logll=true))
#    end
#    GMM(N, data)
end

end