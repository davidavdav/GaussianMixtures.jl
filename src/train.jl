## train.jl  Likelihood calculation and em training for GMMs. 
## (c) 2013--2014 David A. van Leeuwen

using StatsBase: sample 

## Greate a GMM with only one mixture and initialize it to ML parameters
function GMM{T<:FloatingPoint}(x::DataOrMatrix{T}; kind=:diag)
    n, sx, sxx = stats(x, kind=kind)
    μ = sx' ./ n                        # make this a row vector
    d = length(μ)
    if kind == :diag
        Σ = (sxx' - n*μ.*μ) ./ (n-1)
    elseif kind == :full
        ci = cholinv((sxx - n*(μ'*μ)) / (n-1))
        Σ = typeof(ci)[ci]
    else
        error("Unknown kind")
    end
    hist = History(@sprintf("Initlialized single Gaussian d=%d kind=%s with %d data points",
                            d, kind, n))
    GMM(ones(T,1), μ, Σ, [hist], n)
end
## Also allow a Vector, :full makes no sense
GMM{T<:FloatingPoint}(x::Vector{T}) = GMM(x'') # strange idiom...

## constructors based on data or matrix
function GMM{T}(n::Int, x::DataOrMatrix{T}; method::Symbol=:kmeans, kind=:diag,
             nInit::Int=50, nIter::Int=10, nFinal::Int=nIter, sparse=0)
    if n<2
        GMM(x, kind=kind)
    elseif method==:split
        GMM2(n, x, kind=kind, nIter=nIter, nFinal=nFinal, sparse=sparse)
    elseif method==:kmeans
        GMMk(n, x, kind=kind, nInit=nInit, nIter=nIter, sparse=sparse)
    else
        error("Unknown method ", method)
    end
end
## a 1-dimensional Gaussian can be initialized with a vector, skip kind=
GMM{T<:FloatingPoint}(n::Int, x::Vector{T}; method::Symbol=:kmeans, nInit::Int=50, nIter::Int=10, nFinal::Int=nIter, sparse=0) = GMM(n, x''; method=method, kind=:diag, nInit=nInit, nIter=nIter, nFinal=nFinal, sparse=sparse)

## we sometimes end up with pathological gmms
function sanitycheck!(gmm::GMM)
    pathological=NTuple{2}[]
    for i in find(isnan(gmm.μ) | isinf(gmm.μ))
        gmm.μ[i] = 0
        push!(pathological, ind2sub(size(gmm.μ), i))
    end
    if kind(gmm) == :diag
        for i in find(isnan(gmm.Σ) | isinf(gmm.Σ))
            gmm.Σ[i] = 1
            push!(pathological, ind2sub(size(gmm.Σ), i))
        end
    else
        for (si,s) in enumerate(gmm.Σ)
            for i in find(isnan(s) | isinf(s))
                s[i] = 1
                push!(pathological, (si,i))
            end
        end
    end
    if (np=length(pathological)) > 0
        mesg = string(np, " pathological elements normalized")
        addhist!(gmm, mesg)
        warn(mesg)
    end
    pathological
end


## initialize GMM using Clustering.kmeans (which uses a method similar to kmeans++)
function GMMk{T}(n::Int, x::DataOrMatrix{T}; kind=:diag, nInit::Int=50, nIter::Int=10, sparse=0)
    nₓ, d = size(x)
    hist = [History(@sprintf("Initializing GMM, %d Gaussians %s covariance %d dimensions using %d data points", n, diag, d, nₓ))]
    info(last(hist).s)
    ## subsample x to max 1000 points per mean
    nneeded = 1000*n
    if nₓ < nneeded
        if isa(x, Matrix)
            xx = x
        else
            xx = collect(x)             # convert to an array
        end
    else
        if isa(x, Matrix)
            xx = x[sample(1:nₓ, nneeded, replace=false),:]
        else
            ## Data.  Sample an equal amount from every entry in the list x. This reads in 
            ## all data, and may require a lot of memory for very long lists. 
            yy = Matrix[]
            for y in x
                ny = size(y,1)
                nsample = min(ny, @compat ceil(Integer, nneeded / length(x)))
                push!(yy, y[sample(1:ny, nsample, replace=false),:])
            end
            xx = vcat(yy...)
        end
    end
    km = kmeans(xx', n, maxiter=nInit, display = :iter)
    μ::Matrix{T} = km.centers'
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
        Σ = convert(Matrix{T},vcat(map(variance, 1:n)...))
    elseif kind == :full
        function cholinvcov(i::Int)
            sel = km.assignments .== i
            if sum(sel) < d
                return cholinv(eye(d))
            else
                return cholinv(cov(xx[sel,:]))
            end
        end
        Σ = convert(FullCov{T},[cholinvcov(i) for i=1:n])
    else
        error("Unknown kind")
    end
    w::Vector{T} = km.counts ./ sum(km.counts)
    nxx = size(xx,1)
    ng = length(w)
    push!(hist, History(string("K-means with ", nxx, " data points using ", km.iterations, " iterations\n", @sprintf("%3.1f data points per parameter",nxx/((d+1)ng)))))
    info(last(hist).s)
    gmm = GMM(w, μ, Σ, hist, nxx)
    sanitycheck!(gmm)
    em!(gmm, x; nIter=nIter, sparse=sparse)
    gmm
end    

## Train a GMM by consecutively splitting all means.  n most be a power of 2
## This kind of initialization is deterministic, but doesn't work particularily well, its seems
## We start with one Gaussian, and consecutively split.  
function GMM2(n::Int, x::DataOrMatrix; kind=:diag, nIter::Int=10, nFinal::Int=nIter, sparse=0)
    log2n = round(Int,log2(n))
    2^log2n == n || error("n must be power of 2")
    gmm=GMM(x, kind=kind)
    tll = [avll(gmm,x)]
    println("0: avll = ", tll[1])
    for i=1:log2n
        gmm=gmmsplit(gmm)
        avll = em!(gmm, x; nIter=i==log2n ? nFinal : nIter, sparse=sparse)
        println(i, ": avll = ", avll)
        append!(tll, avll)
    end
    println(tll)
    gmm
end

## weighted logsumexp
function logsumexpw(x::Matrix, w::Vector)
    y = x .+ log(w)'
    logsumexp(y, 2)
end

## split a mean according to the covariance matrix
function gmmsplit{T}(μ::Vector{T}, Σ::Matrix{T}, sep=0.2)
    tsep::T = sep
    d, v = eigs(Σ, nev=1)
    p1 = tsep * d[1] * v[:,1]                         # first principal component
    μ - p1, μ + p1
end

function gmmsplit{T}(μ::Vector{T}, Σ::Vector{T}, sep=0.2)
    tsep::T = sep
    maxi = indmax(Σ)
    p1 = zeros(length(μ))
    p1[maxi] = tsep * Σ[maxi]
    μ - p1, μ + p1
end
    
## Split a gmm in order to to double the amount of gaussians
function gmmsplit{T}(gmm::GMM{T}; minweight=1e-5, sep=0.2)
    tsep::T = sep
    ## In this function i, j, and k all index Gaussians
    maxi = reverse(sortperm(gmm.w))
    offInd = find(gmm.w .< minweight)
    if (length(offInd)>0) 
        println("Removing Gaussians with no data");
    end
    for i=1:length(offInd) 
        gmm.w[maxi[i]] = gmm.w[offInd[i]] = gmm.w[maxi[i]]/2;
        gmm.μ[offInd[i],:] = gmm.μ[maxi[i],:] + tsep * √gmm.Σ[maxi[i],:]
        gmm.μ[maxi[i],:] = gmm.μ[maxi[i],:] - tsep * √gmm.Σ[maxi[i],:]
    end
    gmmkind = kind(gmm)
    n = gmm.n
    d = gmm.d
    w = similar(gmm.w, 2n)
    μ = similar(gmm.μ, 2n, d)
    if gmmkind == :diag
        Σ = similar(gmm.Σ, 2n, d)
    else
        Σ = similar(gmm.Σ, 2n)
    end
    for oi=1:n
        ni = 2oi-1 : 2oi
        w[ni] = gmm.w[oi]/2
        if gmmkind == :diag
            μ[ni,:] = hcat(gmmsplit(vec(gmm.μ[oi,:]), vec(gmm.Σ[oi,:]), tsep)...)'
            for k=ni
                Σ[k,:] = gmm.Σ[oi,:]    # implicity copy
            end
        elseif gmmkind == :full
            μ[ni,:] = hcat(gmmsplit(vec(gmm.μ[oi,:]), covar(gmm.Σ[oi]), tsep)...)'
            for k=ni
                Σ[k] = copy(gmm.Σ[oi])
            end
        else
            error("Unknown kind")
        end
    end
    hist = vcat(gmm.hist, History(@sprintf("split to %d Gaussians", 2n)))
    GMM(w, μ, Σ, hist, gmm.nx)
end

# This function runs the Expectation Maximization algorithm on the GMM, and returns
# the log-likelihood history, per data frame per dimension
## Note: 0 iterations is allowed, this just computes the average log likelihood
## of the data and stores this in the history.  
function em!(gmm::GMM, x::DataOrMatrix; nIter::Int = 10, varfloor::Float64=1e-3, sparse=0, debug=1)
    size(x,2)==gmm.d || error("Inconsistent size gmm and x")
    d = gmm.d                   # dim
    ng = gmm.n                  # n gaussians
    initc = gmm.Σ
    ll = zeros(nIter)
    gmmkind = kind(gmm)
    info(string("Running ", nIter, " iterations EM on ", gmmkind, " cov GMM with ", ng, " Gaussians in ", d, " dimensions"))
    for i=1:nIter
        ## E-step
        nₓ, ll[i], N, F, S = stats(gmm, x, parallel=true)
        ## M-step
        gmm.w = N / nₓ
        gmm.μ = F ./ N
        if gmmkind == :diag
            gmm.Σ = S ./ N - gmm.μ.^2
            ## var flooring
            tooSmall = any(gmm.Σ .< varfloor, 2)
            if (any(tooSmall))
                ind = find(tooSmall)
                warn("Variances had to be floored ", join(ind, " "))
                gmm.Σ[ind,:] = initc[ind,:]
            end
        elseif gmmkind == :full
            for k=1:ng
                if N[k] < d
                    warn(@sprintf("Too low occupancy count %3.1f for Gausian %d", N[k], k))
                else
                    μk = gmm.μ[k,:]
                    gmm.Σ[k] = cholinv(S[k] / N[k] - μk' * μk)
                end
            end
        else
            error("Unknown kind")
        end
        sanitycheck!(gmm)
        loginfo = @sprintf("iteration %d, average log likelihood %f",
                           i, ll[i] / (nₓ*d))
        addhist!(gmm, loginfo)
    end
    if nIter>0
        ll /= nₓ * d
        finalll = ll[nIter]
    else
        finalll = avll(gmm, x)
        nₓ = size(x,1)
    end
    gmm.nx = nₓ
    addhist!(gmm,@sprintf("EM with %d data points %d iterations avll %f\n%3.1f data points per parameter",nₓ,nIter,finalll,nₓ/nparams(gmm)))
    ll
end

## this function returns the contributions of the individual Gaussians to the LL
## ll_ij = log p(x_i | gauss_j)
## This is a fast implementation of llpg for diagonal covariance GMMs
## It relies on fast matrix multiplication, and takes up more memory
## TODO: do this the way we do in stats(), which is currently more memory-efficient
function llpg{GT,T<:FloatingPoint}(gmm::GMM{GT,DiagCov{GT}}, x::Matrix{T})
    RT = promote_type(GT,T)
    ## ng = gmm.n
    (nₓ, d) = size(x)
    prec::Matrix{RT} = 1./gmm.Σ         # ng × d
    mp = gmm.μ .* prec                  # mean*precision, ng × d
    ## note that we add exp(-sm2p/2) later to pxx for numerical stability
    normalization = 0.5 * (d * log(2π) .+ sum(log(gmm.Σ),2)) # ng × 1
    sm2p = sum(mp .* gmm.μ, 2)   # sum over d mean^2*precision, ng × 1
    ## from here on data-dependent calculations
    xx = x.^2                           # nₓ × d
    pxx = sm2p' .+ xx * prec'           # nₓ × ng
    mpx = x * mp'                       # nₓ × ng
    # L = broadcast(*, a', exp(mpx-0.5pxx)) # nₓ × ng, Likelihood per frame per Gaussian
    mpx-0.5pxx .- normalization'
end

## A function we see more often... Λ is in chol(inv(Σ)) form
## compute Δ_i = (x_i - μ)' Λ (x_i - μ)
## Note: the return type of Δ should be the promote_type of x and μ/ciΣ
function xμTΛxμ!(Δ::Matrix, x::Matrix, μ::Matrix, ciΣ::UpperTriangular)
    # broadcast!(-, Δ, x, μ)      # size: nₓ × d, add ops: nₓ * d
    (nₓ, d) = size(x)
    @inbounds for j = 1:d
        μj = μ[j]
        for i = 1:nₓ
            Δ[i,j] = x[i,j] - μj
        end
    end
    A_mul_Bc!(Δ, ciΣ)             # size: nₓ × d, mult ops nₓ*d^2
end

## full covariance version of llpg()
function llpg{GT,T<:FloatingPoint}(gmm::GMM{GT,FullCov{GT}}, x::Matrix{T})
    RT = promote_type(GT,T)
    (nₓ, d) = size(x)
    ng = gmm.n
    d==gmm.d || error("Inconsistent size gmm and x")
    ll = Array(RT, nₓ, ng)
    Δ = Array(RT, nₓ, d)
    ## Σ's now are inverse choleski's, so logdet becomes -2sum(log(diag))
    normalization = [0.5d*log(2π) - sum(log(diag((gmm.Σ[k])))) for k=1:ng]
    for k=1:ng
        ## Δ = (x_i - μ_k)' Λ_κ (x_i - m_k)
        xμTΛxμ!(Δ, x, gmm.μ[k,:], gmm.Σ[k])
        ll[:,k] = -0.5sumabs2(Δ,2) .- normalization[k]
    end
    return ll::Matrix{RT}
end
        
## Average log-likelihood per data point and per dimension for a given GMM 
function avll{T<:FloatingPoint}(gmm::GMM, x::Matrix{T})
    gmm.d == size(x,2) || error("Inconsistent size gmm and x")
    mean(logsumexpw(llpg(gmm, x), gmm.w)) / gmm.d
end

## Data version
function avll(gmm::GMM, d::Data)
    llpf = dmap(x->logsumexpw(llpg(gmm,x), gmm.w), d)
    sum(map(sum, llpf)) / sum(map(length, llpf)) / gmm.d
end

## import Distributions.posterior
## this function returns the posterior for component j: p_ij = p(j | gmm, x_i)
## TODO: This is a slow and memory-intensive implementation.  It is better to 
## use the approaches used in stats()
function gmmposterior{GT,T<:FloatingPoint}(gmm::GMM{GT}, x::Matrix{T})      # nₓ × ng
    RT = promote_type(GT,T)
    (nₓ, d) = size(x)
    ng = gmm.n
    d==gmm.d || error("Inconsistent size gmm and x")
    ll = llpg(gmm, x)
    logp = ll .+ log(gmm.w')
    logsump = logsumexp(logp, 2)
    broadcast!(-, logp, logp, logsump)
    exp(logp)::Matrix{RT}, ll::Matrix{RT}
end

