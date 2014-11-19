## gmms.jl  Some functions for a Gaussia Mixture Model
## (c) 2013--2014 David A. van Leeuwen

## uninitialized constructor, defaults to Float64
function GMM(n::Int, d::Int; kind::Symbol=:diag) 
    w = ones(n)/n
    μ = zeros(n, d)
    if kind == :diag
        Σ = ones(n, d)
    elseif kind == :full
        Σ = Matrix{Float64}[eye(d) for i=1:n]
    else
        error("Unknown kind")
    end
    hist = [History(@sprintf "Initialization n=%d, d=%d, kind=%s" n d kind)]
    GMM(w, μ, Σ, hist, 0)
end

Base.eltype{T}(gmm::GMM{T}) = T

## switch between full covariance and inverse cholesky decomposition representations. 
covar{T}(ci::Triangular{T}) = (c = inv(ci); c * c')
cholinv{T}(Σ::Matrix{T}) = chol(inv(cholfact(0.5(Σ+Σ'))), :U)

kind{T}(g::GMM{T,DiagCov{T}}) = :diag
kind{T}(g::GMM{T,FullCov{T}}) = :full

## This may clash with STatsBase
weights(gmm::GMM) = gmm.w
means(gmm::GMM) = gmm.μ
covars{T}(gmm::GMM{T,DiagCov{T}}) = gmm.Σ
covars{T}(gmm::GMM{T,FullCov{T}}) = [covar(ci) for ci in gmm.Σ]

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

function addhist!(gmm::GaussianMixture, s::String) 
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
function Base.full{T}(gmm::GMM{T})
    if kind(gmm) == :full
        return gmm
    end
    Σ = convert(FullCov{T}, [Triangular(diagm(vec(1./√gmm.Σ[i,:])), :U, false) for i=1:gmm.n])
    new = GMM(copy(gmm.w), copy(gmm.μ), Σ, copy(gmm.hist), gmm.nx)
    addhist!(new, "Converted to full covariance")
end

function Base.diag{T}(gmm::GMM{T})
    if kind(gmm) == :diag
        return gmm
    end
    Σ = Array(T, gmm.n, gmm.d)
    for i=1:gmm.n
        Σ[i,:] = 1./abs2(diag(gmm.Σ[i]))
    end
    new = GMM(copy(gmm.w), copy(gmm.μ), Σ, copy(gmm.hist), gmm.nx)
    addhist!(new, "Converted to diag covariance")
end

function history(gmm::GaussianMixture) 
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
function Base.show{T}(io::IO, gmm::GMM{T})
    println(io, @sprintf("GMM{%s} with %d components in %d dimensions and %s covariance", T, gmm.n, gmm.d, kind(gmm)))
    gmmkind = kind(gmm)
    for j=1:gmm.n
        println(io, @sprintf "Mix %d: weight %f" j gmm.w[j]);
        println(io, "mean: ", gmm.μ[j,:])
        if gmmkind == :diag
            println(io, "variance: ", gmm.Σ[j,:])
        elseif gmmkind == :full
            println(io, "covariance: ", covar(gmm.Σ[j]))
        else
            printf("Unknown kind")
        end
    end
end

## some routines for conversion between float types
for (f,t) in ((:float16, Float16), (:float32, Float32), (:float64, Float64))
    eval(Expr(:import, :Base, f))
    @eval begin
        function Base.convert{T}(::Type{GMM{$t}}, gmm::GMM{T})
            T == $t && return gmm
            h = vcat(gmm.hist, History(string("Converted to ", $t)))
            w = ($f)(gmm.w)
            μ = ($f)(gmm.μ)
            gmmkind = kind(gmm)
            if gmmkind == :full
                Σ = eltype(FullCov{$t})[($f)(x) for x in gmm.Σ]
            elseif gmmkind == :diag
                Σ = ($f)(gmm.Σ)
            else
                error("Unknown kind")
            end
            GMM(w, μ, Σ, h, gmm.nx)
        end
        function Base.convert{T}(::Type{VGMM{$t}}, vg::VGMM{T})
            T == $t && return vg
            h = vcat(vg.hist, History(string("Converted to ", $t)))
            W = map($f, vg.W)
            VGMM(vg.n, vg.d, ($f)(vg.π), ($f)(vg.α), ($f)(vg.β), ($f)(vg.m),
                 ($f)(vg.ν), W, h)
        end
        function Base.convert{T}(::Type{GMMprior{$t}}, p::GMMprior{T})
            T == $t && return p
            GMMprior(($f)(p.α0), ($f)(p.β0), ($f)(p.m0), ($f)(p.ν0), ($f)(p.W0))
        end
    end
    for T in (GMM, VGMM, GMMprior)
        @eval ($f)(x::$T) = convert($T{$t}, x)
    end
end

## conversion to MixtureModel
function Distributions.MixtureModel(gmm::GMM{Float64})
    if kind(gmm) == :full
        mixtures = [MvNormal(vec(gmm.μ[i,:]), covar(gmm.Σ[i])) for i=1:gmm.n]
    else
        mixtures = [MvNormal(vec(gmm.μ[i,:]), vec(gmm.Σ[i,:])) for i=1:gmm.n]
    end
    MixtureModel(mixtures, gmm.w)
end

## conversion to GMM
function GMM(m::MixtureModel{Multivariate,Continuous,MvNormal{PDMat,Vector{Float64}}})
    Σ = eltype(FullCov{Float64})[cholinv(c.Σ.mat) for c in m.components]
    μ = hcat([c.μ for c in m.components]...)'
    w = m.prior.prob
    n, d = size(μ)
    h = [History(@sprintf("Initialization from MixtureModel n=%d, d=%d, kind=full", n, d))]
    GMM(w, μ, Σ, h, 0)
end

function GMM(m::MixtureModel{Multivariate,Continuous,MvNormal{PDiagMat,Vector{Float64}}})
    Σ = hcat([c.Σ.diag for c in m.components]...)'
    μ = hcat([c.μ for c in m.components]...)'
    w = m.prior.prob
    n, d = size(μ)
    h = [History(@sprintf("Initialization from MixtureModel n=%d, d=%d, kind=diag", n, d))]
    GMM(w, μ, Σ, h, 0)
end
