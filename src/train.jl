## train.jl  Likelihood calculation and em training for GMMs. 
## (c) 2013--2014 David A. van Leeuwen


## Greate a GMM with only one mixture and initialize it to ML parameters
function GMM(x::DataOrMatrix; kind=:diag)
    numtype = eltype(x)
    n, sx, sxx = stats(x, kind=kind)
    μ = sx' ./ n                        # make this a row vector
    d = length(μ)
    if kind == :diag
        Σ = (sxx' - n*μ.*μ) ./ (n-1)
    else
        Σ =  Matrix{numtype}[(sxx - n*(μ'*μ)) ./ (n-1)]
    end
    hist = History(@sprintf("Initlialized single Gaussian d=%d kind=%s with %d data points",
                            d, kind, n))
    GMM(kind, numtype[1.0], μ, Σ, [hist])
end
## Also allow a Vector
GMM{T<:Real}(x::Vector{T}) = GMM(x'')

## constructors based on data or matrix
function GMM(n::Int, x::DataOrMatrix, method::Symbol=:kmeans; kind=:diag, nInit::Int=50, nIter::Int=10, nFinal::Int=nIter)
    if n<2
        GMM(x, kind=kind)
    elseif method==:split
        GMM2(n, x, kind=kind, nIter=nIter, nFinal=nFinal)
    elseif method==:kmeans
        GMMk(n, x, kind=kind, nInit=nInit, nIter=nIter)
    else
        error("Unknown method ", method)
    end
end
## a 1-dimensional Gaussian can bi initialized with a vector
GMM{T<:FloatingPoint}(n::Int, x::Vector{T}, method::Symbol=:kmeans; kind=:diag, nInit::Int=50, nIter::Int=10, nFinal::Int=nIter) = GMM(n, x'', method;  kind=kind, nInit=nInit, nIter=nIter, nFinal=nFinal)

## initialize GMM using Clustering.kmeans (which uses a method similar to kmeans++)
function GMMk(n::Int, x::DataOrMatrix; kind=:diag, nInit::Int=50, nIter::Int=10)
    nx, d = size(x)
    ## subsample x to max 1000 points per mean
    nneeded = 1000*n
    if nx < nneeded
        if isa(x, Matrix)
            xx = x
        else
            xx = collect(x)             # convert to an array
        end
    else
        if isa(x, Matrix)
            xx = x[sample(1:nx, nneeded, replace=false),:]
        else
            nsample = iceil(nneeded / length(x))
            xx = vcat([y[sample(1:size(y,1), nsample, replace=false),:] for y in x]...)
        end
    end
    km = kmeans(float64(xx'), n, maxiter=nInit, display = :iter)
    μ = km.centers'
    if kind == :diag
        ## helper that deals with centers with singleton datapoints.
        function variance(i::Int)
            sel = km.assignments .== i
            if length(sel) < 2
                return ones(1,d)
            else 
                return var(xx[sel,:],1)                
            end
        end
        Σ = convert(Matrix{Float64},vcat(map(variance, 1:n)...))
    else
        function covariance(i::Int)
            sel = km.assignments .== i
            if length(sel) < 2
                return eye(d)
            else
                return cov(xx[sel,:])
            end
        end
        Σ = Matrix{Float64}[covariance(i) for i=1:n]
        println(Σ)
    end
    w = km.counts ./ sum(km.counts)
    hist = History(string("K-means with ", size(xx,1), " data points using ", km.iterations, " iterations\n"))
    gmm = GMM(kind, w, μ, Σ, [hist])
    addhist!(gmm, @sprintf("%3.1f data points per parameter",nx/nparams(gmm)))
    em!(gmm, x; nIter=nIter)
    gmm
end    

## Train a GMM by consecutively splitting all means.  n most be a power of 2
## This kind of initialization is deterministic, but doesn't work particularily well, its seems
## We start with one Gaussian, and consecutively split.  
function GMM2(n::Int, x::DataOrMatrix; kind=:diag, nIter::Int=10, nFinal::Int=nIter)
    log2n = int(log2(n))
    @assert 2^log2n == n
    gmm=GMM(x, kind=kind)
    tll = avll(gmm,x)
    println("0: avll = ", tll)
    for i=1:log2n
        gmm=split(gmm)
        avll = em!(gmm, x; nIter=i==log2n ? nFinal : nIter)
        println(i, ": avll = ", avll)
        tll = vcat(tll, avll)
    end
    println(tll)
    gmm
end

## weighted logsumexp
function logsumexpw(x::Matrix, w::Vector)
    y = broadcast(+, x, log(w)')
    logsumexp(y, 2)
end

import Base.split
## split a mean according to the covariance matrix
function split(μ::Vector, Σ::Matrix, sep::Float64=0.2)
    d, v = eigs(Σ, nev=1)
    p1 = sep * d[1] * v[:,1]                         # first principal component
    μ - p1, μ + p1
end

function split(μ::Vector, Σ::Vector, sep::Float64=0.2)
    maxi = indmax(Σ)
    p1 = zeros(length(μ))
    p1[maxi] = sep * Σ[maxi]
    μ - p1, μ + p1
end
    
## Split a gmm in order to to double the amount of gaussians
function split(gmm::GMM; minweight::Float64=1e-5, sep::Float64=0.2)
    ## In this function i, j, and k all index Gaussians
    maxi = reverse(sortperm(gmm.w))
    offInd = find(gmm.w .< minweight)
    if (length(offInd)>0) 
        println("Removing Gaussians with no data");
    end
    for i=1:length(offInd) 
        gmm.w[maxi[i]] = gmm.w[offInd[i]] = gmm.w[maxi[i]]/2;
        gmm.μ[offInd[i],:] = gmm.μ[maxi[i],:] + sep*sqrt((gmm.Σ[maxi[i],:]))
        gmm.μ[maxi[i],:] = gmm.μ[maxi[i],:] - sep*sqrt((gmm.Σ[maxi[i],:]))
    end
    new = GMM(2gmm.n, gmm.d, kind=gmm.kind)
    for oi=1:gmm.n
        ni = 2oi-1 : 2oi
        new.w[ni] = gmm.w[oi]/2
        if gmm.kind == :diag
            new.μ[ni,:] = hcat(split(vec(gmm.μ[oi,:]), vec(gmm.Σ[oi,:]), sep)...)'
            for k=ni
                new.Σ[k,:] = gmm.Σ[oi,:]
            end
        else
            new.μ[ni,:] = hcat(split(vec(gmm.μ[oi,:]), gmm.Σ[oi], sep)...)'
            for k=ni
                new.Σ[k] = gmm.Σ[oi]
            end
        end
    end
    new.hist = vcat(gmm.hist, History(@sprintf("split to %d Gaussians",new.n)))
    new
end

# This function runs the Expectation Maximization algorithm on the GMM, and returns
# the log-likelihood history, per data frame per dimension
## Note: 0 iterations is allowed, this just computes the average log likelihood
## of the data and stores this in the history.  
function em!(gmm::GMM, x::DataOrMatrix; nIter::Int = 10, varfloor::Float64=1e-3)
    @assert size(x,2)==gmm.d
    d = gmm.d                   # dim
    ng = gmm.n                  # n gaussians
    initc = gmm.Σ
    ll = zeros(nIter)
    for i=1:nIter
        ## E-step
        nx, ll[i], N, F, S = stats(gmm, x, parallel=true)
        ## M-step
        gmm.w = N / nx
        gmm.μ = broadcast(/, F, N)
        if gmm.kind == :diag
            gmm.Σ = broadcast(/, S, N) - gmm.μ.^2
            ## var flooring
            tooSmall = any(gmm.Σ .< varfloor, 2)
            if (any(tooSmall))
                ind = find(tooSmall)
                println("Variances had to be floored ", join(ind, " "))
                gmm.Σ[ind,:] = initc[ind,:]
            end
        elseif gmm.kind == :full
            for i=1:ng
                μi = gmm.μ[i,:]
                gmm.Σ[i][:] = S[i] / N[i] - μi' * μi
            end
        end
    end
    if nIter>0
        ll /= nx * d
        finalll = ll[nIter]
    else
        finalll = avll(gmm, x)
        nx = size(x,1)
    end
    addhist!(gmm,@sprintf("EM with %d data points %d iterations avll %f\n%3.1f data points per parameter",nx,nIter,finalll,size(x,1)/nparams(gmm)))
    ll
end

## This is a fast implementation of llpg for diagonal covariance GMMs
## It relies on fast matrix multiplication, and takes up more memory
function llpgdiag(gmm::GMM, x::Matrix)
    ## ng = gmm.n
    (nx, d) = size(x)
    prec = 1./gmm.Σ                     # ng * d
    mp = gmm.μ .* prec                  # mean*precision, ng * d
    ## note that we add exp(-sm2p/2) later to pxx for numerical stability
    normalization = 0.5 * (d * log(2π) .+ sum(log(gmm.Σ),2)) # ng * 1
    sm2p = sum(mp .* gmm.μ, 2)   # sum over d mean^2*precision, ng * 1
    xx = x.^2                    # nx * d
    pxx = broadcast(+, sm2p', xx * prec') # nx * ng
    mpx = x * mp'                         # nx * ng
    # L = broadcast(*, a', exp(mpx-0.5pxx)) # nx * ng, Likelihood per frame per Gaussian
    broadcast(-, mpx-0.5pxx, normalization')
end

## this function returns the contributions of the individual Gaussians to the LL
## ll_ij = log p(x_i | gauss_j)
function llpg{T<:FloatingPoint}(gmm::GMM, x::Matrix{T})
    (nx, d) = size(x)
    ng = gmm.n
    @assert d==gmm.d    
    if (gmm.kind==:diag)
        return llpgdiag(gmm, x)
        ## old, slow code
        ll = Array(Float64, nx, ng)
        normalization = 0.5 * (d * log(2π) .+ sum(log(gmm.Σ),2)) # row 1...ng
        for j=1:ng
            Δ = broadcast(-, x, gmm.μ[j,:]) # nx * d
            ΔsqrtΛ = broadcast(/, Δ, sqrt(gmm.Σ[j,:]))
            ll[:,j] = -0.5sumsq(ΔsqrtΛ,2) .- normalization[j]
        end
        return ll
    else
        ll = Array(Float64, nx, ng)
        # C = [chol(inv(gmm.Σ[i]), :L) for i=1:ng] # should do this only once...
        normalization = 0.5 * [d*log(2π) + logdet(gmm.Σ[i]) for i=1:ng]
        for j=1:ng
            C = chol(inv(gmm.Σ[j]), :L)
            Δ = broadcast(-, x, gmm.μ[j,:]) # nx * d
            CΔ = Δ*C                 # nx * d, nx*d^2 operations
            ll[:,j] = -0.5sumsq(CΔ,2) .- normalization[j]
        end
        return ll
    end
end
        
## Average log-likelihood per data point and per dimension for a given GMM 
function avll{T<:FloatingPoint}(gmm::GMM, x::Matrix{T})
    @assert gmm.d == size(x,2)
    mean(logsumexpw(llpg(gmm, x), gmm.w)) / gmm.d
end

## Data version
function avll(gmm::GMM, d::Data)
    llpf = map(x->logsumexpw(llpg(gmm,x), gmm.w), d)
    sum(map(sum, llpf)) / sum(map(length, llpf)) / gmm.d
end

import Distributions.posterior
## this function returns the posterior for component j: p_ij = p(j | gmm, x_i)
function posterior{T}(gmm, x::Matrix{T}, getll=false)      # nx * ng
    (nx, d) = size(x)
    ng = gmm.n
    @assert d==gmm.d
    ll = llpg(gmm, x)
    logp = broadcast(+, ll, log(gmm.w'))
    logsump = logsumexp(logp, 2)
    broadcast!(-, logp, logp, logsump)
    if getll
        exp(logp), ll
    else
        exp(logp)
    end
end

