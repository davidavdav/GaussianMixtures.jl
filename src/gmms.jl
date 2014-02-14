## gmms.jl  Some functions for a Gaussia Mixture Model
## (c) 2013--2014 David A. van Leeuwen

## some init code.  Turn off subnormal computation, as it is slow.  This is a global setting...
ccall(:jl_zero_subnormals, Bool, (Bool,), true)

#require("gmmtypes.jl")

nparams(gmm::GMM) = sum(map(length, (gmm.w, gmm.μ, gmm.Σ)))
weights(gmm::GMM) = gmm.w
means(gmm::GMM) = gmm.μ
covars(gmm::GMM) = gmm.Σ

using Clustering

import Base.copy

# call me old-fashioned
#nrow(x) = size(x,1)
#ncol(x) = size(x,2)

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
function GMM{T<:FloatingPoint}(x::Array{T,2})
    d=size(x, 2)
    gmm = GMM(1,d)
    gmm.μ = mean(x, 1)
    gmm.Σ = var(x, 1)
    addhist!(gmm, @sprintf("Initlialized single Gaussian with %d data points",size(x,1)))
    gmm
end

## Same, but initialize using type Data, possibly doing things in parallel in stats()
function GMM(x::Data)
    n, sx, sxx = stats(x)
    μ = sx ./ n
    Σ = (sxx - n*μ.^2) ./ (n-1)
    d = length(μ)
    gmm = GMM(1,d)
    gmm.μ = μ'
    gmm.Σ = Σ'
    addhist!(gmm, @sprintf("Initlialized single Gaussian with %d data points", n))
    gmm
end

function GMM{T<:FloatingPoint}(n::Int, x::DataOrMatrix{T}, method::Symbol=:kmeans; nInit::Int=50, nIter::Int=10, nFinal::Int=nIter, fast=true, logll=true)
    if method==:split
        GMM2(n, x, nIter=nIter, nFinal=nFinal, fast=fast, logll=logll)
    elseif method==:kmeans
        GMMk(n, x, nInit=nInit, nIter=nIter)
    else
        error("Unknown method ", method)
    end
end

## initialize GMM using Clustering.kmeans (which uses a method similar to kmeans++)
function GMMk{T<:FloatingPoint}(n::Int, x::DataOrMatrix{T}; nInit::Int=50, nIter::Int=10, logll=true)
    gmm = GMM(n, ncol(x))
    km = kmeans(convert(Array{Float64},x'), n, max_iter=nInit, display = logll ? :iter : :none)
    gmm.μ = km.centers'
    ## helper that deals with centers with singleton datapoints. 
    function variance(i::Int)
        sel = km.assignments .== i
        if length(i)<2
            return ones(1,ncol(x))
        end
        return var(x[sel,:],1)
    end
    gmm.Σ = convert(Array{Float64,2},vcat(map(variance, 1:n)...))
    gmm.w = km.counts ./ sum(km.counts)
    addhist!(gmm, string("K-means with ", size(x,1), " data points using ", km.iterations, " iterations\n", @sprintf("%3.1f data points per parameter",size(x,1)/nparams(gmm))))
    em!(gmm, x; nIter=nIter, logll=logll)
    gmm
end    

## Train a GMM by consecutively splitting all means.  n most be a power of 2
## This kind of initialization is deterministic, but doesn't work particularily well, its seems
## We start with one Gaussian, and consecutively split.  
function GMM2{T<:FloatingPoint}(n::Int, x::DataOrMatrix{T}; nIter::Int=10, nFinal::Int=nIter, fast=true, logll=true)
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
GMM{T<:FloatingPoint}(n::Int,x::Vector{T};nIter::Int=10) = GMM(n, reshape(x, length(x), 1);  nIter=nIter)

## Average log-likelihood per data point and per dimension for a given GMM 
function avll{T<:FloatingPoint}(gmm::GMM, x::Array{T,2})
    @assert gmm.d == size(x,2)
    llpfpg = llpg(gmm, x)
    llpf = log(exp(llpfpg) * gmm.w)     # this possibly loses some accuracy
    mean(llpf) / gmm.d
end

## Data version
function avll(gmm::GMM, d::Data)
    llpf = map(x->log(exp(llpg(gmm,x)) * gmm.w), d)
    sum(map(sum, llpf)) / sum(map(length, llpf)) / gmm.d
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
    new.hist = vcat(gmm.hist, History(@sprintf("split to %d Gaussians",new.n)))
    new
end

# This function runs the Expectation Maximization algorithm on the GMM, and returns
# the log-likelihood history, per data frame per dimension
## Note: 0 iterations is allowed, this just computes the average log likelihood
## of the data and stores this in the history.  
function em!{T<:FloatingPoint}(gmm::GMM, x::DataOrMatrix{T}; nIter::Int = 10, varfloor::Real=1e-3, logll=true, fast=true)
    @assert size(x,2)==gmm.d
    MEM = mem*(2<<30)           # now a parameter
    d = gmm.d                   # dim
    ng = gmm.n                  # n gaussians
    initc = gmm.Σ
    blocksize = floor(MEM/((3+3ng)sizeof(Float64))) # 3 instances of nx*ng
    nf = size(x, 1)             # n frames
    ll = zeros(nIter)
    nx = 0
    for i=1:nIter
        ## E-step
        if fast
            nx, ll[i], N, F, S = stats(gmm, x, parallel=true)
        else
            b = 0                  # pointer to start
            N = zeros(ng)
            S = zeros(ng,d)
            F = zeros(ng,d)
            while (b < nf) 
                e=min(b+blocksize, nf)
                xx = x[b+1:e,:]
                nxx = e-b
                (p,a) = post(gmm, xx) # nx * ng
                N += sum(p,1)'
                F += p' * xx
                S += p' * xx.^2
                if (logll || i==nIter) 
                    ll[i] += sum(log(a*gmm.w))
                end
            end
            b += nxx             # b=e
            nx = b
        end
        ## M-step
        gmm.w = N / nx
        gmm.μ = broadcast(/, F, N)
        gmm.Σ = broadcast(/, S, N) - gmm.μ.^2
        ## var flooring
        tooSmall = any(gmm.Σ .< varfloor, 2)
        if (any(tooSmall))
            ind = find(tooSmall)
            println("Variances had to be floored ", join(ind, " "))
            gmm.Σ[ind,:] = initc[ind,:]
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

## this function returns the contributions of the individual Gaussians to the LL
## ll_ij = log p(x_i | gauss_j)
function llpg{T<:FloatingPoint}(gmm::GMM, x::Array{T,2})
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
        s = split(h.s, "\n")
        print(@sprintf("%6.3f\t%s\n", h.t-t0, s[1]))
        for i=2:length(s) 
            print(@sprintf("%6s\t%s\n", " ", s[i]))
        end
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

