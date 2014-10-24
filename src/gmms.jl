## gmms.jl  Some functions for a Gaussia Mixture Model
## (c) 2013--2014 David A. van Leeuwen

#require("gmmtypes.jl")

## uninitialized constructor, defaults to Float64
function GMM(n::Int, d::Int; kind::Symbol=:diag) 
    w = ones(n)/n
    μ = zeros(n, d)
    if kind == :diag
        Σ = ones(n, d)
    else
        Σ = Matrix{Float64}[eye(d) for i=1:n]
    end
    hist = [History(@sprintf "Initialization n=%d, d=%d, kind=%s" n d kind)]
    GMM(kind, w, μ, Σ, hist)
end

## initialized constructor, outdated?
function GMM(weights::Vector, means::Array, covars::Array, kind::Symbol)
    n = length(weights)
    d = size(means,2)
    hist = [History(@sprintf "Initialized from weights, means, covars; n=%d, d=%d, kind=%s" n d kind)]
    GMM(kind, weights, means, covars, hist)
end

function nparams(gmm::GMM)
    if gmm.kind==:diag
        sum(map(length, (gmm.w, gmm.μ, gmm.Σ)))-1
    else
        sum(map(length, (gmm.w, gmm.μ))) - 1 + gmm.n * gmm.d * (gmm.d+1) / 2
    end
end
weights(gmm::GMM) = gmm.w
means(gmm::GMM) = gmm.μ
covars(gmm::GMM) = gmm.Σ

function addhist!(gmm::GMM, s::String) 
    gmm.hist = vcat(gmm.hist, History(s))
end

## copy a GMM, deep-copying all internal information. 
function Base.copy(gmm::GMM)
    w = copy(gmm.w)
    μ = copy(gmm.μ)
    Σ = deepcopy(gmm.Σ)
    hist = copy(gmm.hist)
    g = GMM(gmm.kind, w, μ, Σ, hist)
    addhist!(g, "Copy")
    g
end

## create a full cov GMM from a diag cov GMM (for testing full covariance routines)
function Base.full(gmm::GMM)
    if gmm.kind == :full
        return gmm
    end
    Σ = Matrix{Float64}[diagm(vec(gmm.Σ[i,:])) for i=1:gmm.n]
    new = GMM(:full, copy(gmm.w), copy(gmm.μ), Σ, copy(gmm.hist))
    addhist!(new, "Converted to full covariance")
    new
end

function Base.diag{T}(gmm::GMM{T})
    if gmm.kind == :diag
        return gmm
    end
    Σ = Array(T, gmm.n, gmm.d)
    for i=1:gmm.n
        Σ[i,:] = diag(gmm.Σ[i])
    end
    new = GMM(:diag, copy(gmm.w), copy(gmm.μ), Σ, copy(gmm.hist))
    addhist!(new, "Converted to diag covariance")
    new
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

## we could improve this a lot
function Base.show(io::IO, gmm::GMM) 
    println(io, @sprintf "GMM with %d components in %d dimensions and %s covariance" gmm.n gmm.d gmm.kind)
    for j=1:gmm.n
        println(io, @sprintf "Mix %d: weight %f" j gmm.w[j]);
        println(io, "mean: ", gmm.μ[j,:])
        if gmm.kind == :diag
            println(io, "variance: ", gmm.Σ[j,:])
        elseif gmm.kind == :full
            println(io, "covariance: ", gmm.Σ[j])
        end
    end
end

Base.eltype(gmm::GMM) = eltype(gmm.w)

## some conversion routines
for (f,t) in ((:float32, Float32), (:float64, Float64))
    eval(Expr(:import, :Base, f))
    @eval function ($f)(gmm::GMM)
        h = vcat(gmm.hist, History(string("Converted to ", $t)))
        w = ($f)(gmm.w)
        μ = ($f)(gmm.μ)
        if gmm.kind == :full
            Σ = Matrix{$t}[($f)(x) for x in gmm.Σ]
        else
            Σ = ($f)(gmm.Σ)
        end
        GMM(gmm.kind, w, μ, Σ, h)
    end
end

        
