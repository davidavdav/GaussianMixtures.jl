## gmms.jl  Some functions for a Gaussia Mixture Model
## (c) 2013--2014 David A. van Leeuwen

import LinearAlgebra.AbstractTriangular
using Logging

## uninitialized constructor, defaults to Float64
"""
`GMM(n::Int, d::Int, kind::Symbol=:diag)` initializes a GMM with means 0 and Indentity covariances
"""
function GMM(n::Int, d::Int; kind::Symbol=:diag)
    w = ones(n)/n
    μ = zeros(n, d)
    if kind == :diag
        Σ = ones(n, d)
    elseif kind == :full
        Σ = UpperTriangular{Float64, Matrix{Float64}}[UpperTriangular(eye(d)) for i=1:n]
    else
        error("Unknown kind")
    end
    hist = [History(@sprintf "Initialization n=%d, d=%d, kind=%s" n d kind)]
    GMM(w, μ, Σ, hist, 0)
end

Base.eltype(gmm::GMM{T}) where {T} = T

## switch between full covariance and inverse cholesky decomposition representations.
"""
`covar(GMM.Σ)` extracts the covariances Σ (which may be encoded as chol(inv(Σ))
"""
covar(ci::AbstractTriangular{T}) where {T} = (c = inv(ci); c * c')
cholinv(Σ::Matrix{T}) where {T} = cholesky(inv(cholesky(0.5(Σ+Σ')))).U

"""
`kind(::GMM)` returns the kind of GMM, either `:diag` or `:full`
"""
function kind(g::GMM{T, CT}) where {T, CT <: DiagCov}
    return :diag
end
function kind(g::GMM{T, CT}) where {T, CT <: FullCov}
    return :full
end

## This may clash with STatsBase
"""
`weights(::GMM)` returns the weights `w`, or priors, of the Gaussians in the mixture
"""
weights(gmm::GMM) = gmm.w
"`means(::GMM)` returns the means `μ` of the Gaussians in the mixture"
means(gmm::GMM) = gmm.μ
"`covars(::GMM)` returns the covariance matrices Σ of the Gaussians in the mixture."
covars(gmm::GMM{T,<:DiagCov{T}}) where {T} = gmm.Σ
covars(gmm::GMM{T,FullCov{T}}) where {T} = [covar(ci) for ci in gmm.Σ]

"`nparams(::GMM)` returns the number of free parameters in the GMM"
function nparams(gmm::GMM)
    gmmkind = kind(gmm)
    if gmmkind ==:diag
        gmm.n * (1 + 2gmm.d) - 1
#        sum(map(length, (gmm.w, gmm.μ, gmm.Σ)))-1
    elseif gmmkind == :full
        gmm.n * (1 + gmm.d + div((gmm.d+1)gmm.d, 2)) - 1
#        sum(map(length, (gmm.w, gmm.μ))) - 1 + div(gmm.n * gmm.d * (gmm.d+1), 2)
    else
        error("Unknown kind")
    end
end

"`addhist!(::GMM, s)` adds a comment `s` to the GMMM"
function addhist!(gmm::GaussianMixture, s::AbstractString)
    @logmsg moreInfo s
    push!(gmm.hist, History(s))
    gmm
end

## copy a GMM, deep-copying all internal information.
function Base.copy(gmm::GMM)
    w = copy(gmm.w)
    μ = copy(gmm.μ)
    Σ = deepcopy(gmm.Σ)
    hist = copy(gmm.hist)
    g = GMM(w, μ, Σ, hist, gmm.nx)
    addhist!(g, "Copy")
end

## create a full cov GMM from a diag cov GMM (for testing full covariance routines)
"`full(::GMM)` turns a diagonal covariance GMM into a full-covariance GMMM"
function full(gmm::GMM{T}) where {T}
    if kind(gmm) == :full
        return gmm
    end
    Σ = convert(FullCov{T}, [UpperTriangular(diagm(vec(1 ./√gmm.Σ[i,:]))) for i=1:gmm.n])
    new = GMM(copy(gmm.w), copy(gmm.μ), Σ, copy(gmm.hist), gmm.nx)
    addhist!(new, "Converted to full covariance")
end

"""`diag(::GMM)` turns a full-covariance GMM into a diagonal-covariance GMM, by ignoring
off-diagonal elements"""
function LinearAlgebra.diag(gmm::GMM{T}) where {T}
    if kind(gmm) == :diag
        return gmm
    end
    Σ = Array{T}(gmm.n, gmm.d)
    for i=1:gmm.n
        Σ[i,:] = 1 ./ abs2(diag(gmm.Σ[i]))
    end
    new = GMM(copy(gmm.w), copy(gmm.μ), Σ, copy(gmm.hist), gmm.nx)
    addhist!(new, "Converted to diag covariance")
end

function Base.show(io::IO, h::History)
    println(io, Libc.strftime(h.t), ": ", h.s)
end

history(gmm::GaussianMixture) = gmm.hist

function Base.show(io::IO, ::MIME"text/plain", hist::Vector{History})
    t0 = hist[1].t
    println(io, "GMM trained from ", Libc.strftime(t0), " to ", Libc.strftime(last(hist).t))
    for h in hist
        s = split(h.s, "\n")
        print(io, @sprintf("%6.3f\t%s\n", h.t-t0, s[1]))
        for i = 2:length(s)
            print(io, @sprintf("%6s\t%s\n", " ", s[i]))
        end
    end
end

# compute the ranges to be displayed, plus a total index comprising all ranges.
function compute_range(maxn, n)
    if maxn < n
        hn = div(maxn,2)
        r = (1:hn, n-hn+1:n)
    else
        r = (1:n,)
    end
    totr = vcat(map(collect, r)...)
    r, totr
end

## we could improve this a lot
function Base.show(io::IO, mime::MIME"text/plain", gmm::GMM{T}) where {T}
    println(io, @sprintf("GMM{%s} with %d components in %d dimensions and %s covariance", T, gmm.n, gmm.d, kind(gmm)))
    gmmkind = kind(gmm)
    if gmmkind == :diag
        maxngauss = clamp((displaysize(io)[1] - 1) ÷ 6gmm.n, 1, gmm.n)
        ranges, = compute_range(maxngauss, gmm.n)
        for (i, r) in enumerate(ranges)
            if i > 1
                println(io, "⋮")
            end
            for j in r
                println(io, @sprintf "Mix %d: weight %f" j gmm.w[j]);
                println(io, "  mean: ", gmm.μ[j,:])
                println(io, "  variance: ", gmm.Σ[j,:])
            end
        end
    elseif gmmkind == :full
        nlinesneeded = gmm.n * (4 + gmm.d) + 1
        if displaysize(io)[1] > nlinesneeded
            ranges = (1:gmm.n,)
        else
            maxngauss = clamp((displaysize(io)[1] - 1) ÷ (4 + gmm.d), min(gmm.n,3), gmm.n)
            ranges, = compute_range(maxngauss, gmm.n)
            println(maxngauss, " ", ranges)
        end
        for (i, r) in enumerate(ranges)
            if i > 1
                println(io, "⋮")
            end
            for j in r
                println(io, @sprintf "Mix %d: weight %f" j gmm.w[j]);
                println(io, " mean: ", gmm.μ[j,:])
                print(io, " covariance: ")
                show(io, mime, covar(gmm.Σ[j]))
                println(io)
            end
        end
    else
        printf("Unknown kind")
    end
end

## some routines for conversion between float types
#    @doc """`convert(GMM{::Type}, GMM)` convert the GMM to a different floating point type""" ->
function Base.convert(::Type{GMM{Td, Cd}}, gmm::GMM{Ts, Cs}) where {Td,Cd,Ts,Cs}
    (Ts == Td) && (Cs == Cd) && return gmm
    h = vcat(gmm.hist, History(string("Converted to ", Td)))
    w = map(Td, gmm.w)
    μ = map(Td, gmm.μ)
    Σ = map(eltype(Cd),  gmm.Σ)
    return GMM{Td,Cd}(w, μ, Σ, h, gmm.nx)
end
function Base.convert(::Type{VGMM{Td}}, vg::VGMM{Ts}) where {Td,Ts}
    Ts == Td && return vg
    h = vcat(vg.hist, History(string("Converted to ", Td)))
    W = map(eltype(FullCov{Td}), vg.W)
    π = convert(GMMprior{Td}, vg.π)
    VGMM(vg.n, vg.d, π, map(Td,vg.α), map(Td, vg.β), map(Td,vg.m),
    map(Td, vg.ν), W, h)
end
function Base.convert(::Type{GMMprior{Td}}, p::GMMprior{Ts}) where {Td,Ts}
    Ts == Td && return p
    GMMprior(map(Td, p.α₀), map(Td, p.β₀), map(Td, p.m₀), map(Td, p.ν₀), map(Td, p.W₀))
end
