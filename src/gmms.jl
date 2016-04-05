## gmms.jl  Some functions for a Gaussia Mixture Model
## (c) 2013--2014 David A. van Leeuwen

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
        Σ = Matrix{Float64}[eye(d) for i=1:n]
    else
        error("Unknown kind")
    end
    hist = [History(@sprintf "Initialization n=%d, d=%d, kind=%s" n d kind)]
    GMM(w, μ, Σ, hist, 0)
end

Base.eltype{T}(gmm::GMM{T}) = T

## switch between full covariance and inverse cholesky decomposition representations.
"""
`covar(GMM.Σ)` extracts the covariances Σ (which may be encoded as chol(inv(Σ))
"""
covar{T}(ci::AbstractTriangular{T}) = (c = inv(ci); c * c')
cholinv{T}(Σ::Matrix{T}) = chol(inv(cholfact(0.5(Σ+Σ'))), Val{:U})

"""
`kind(::GMM)` returns the kind of GMM, either `:diag` or `:full`
"""
kind{T}(g::GMM{T,DiagCov{T}}) = :diag
kind{T}(g::GMM{T,FullCov{T}}) = :full

## This may clash with STatsBase
"""
`weights(::GMM)` returns the weights `w`, or priors, of the Gaussians in the mixture
"""
weights(gmm::GMM) = gmm.w
"`means(::GMM)` returns the means `μ` of the Gaussians in the mixture"
means(gmm::GMM) = gmm.μ
"`covars(::GMM)` returns the covariance matrices Σ of the Gaussians in the mixture."
covars{T}(gmm::GMM{T,DiagCov{T}}) = gmm.Σ
covars{T}(gmm::GMM{T,FullCov{T}}) = [covar(ci) for ci in gmm.Σ]

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
    info(s)
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
function Base.full{T}(gmm::GMM{T})
    if kind(gmm) == :full
        return gmm
    end
    Σ = convert(FullCov{T}, [UpperTriangular(diagm(vec(1./√gmm.Σ[i,:]))) for i=1:gmm.n])
    new = GMM(copy(gmm.w), copy(gmm.μ), Σ, copy(gmm.hist), gmm.nx)
    addhist!(new, "Converted to full covariance")
end

"""`diag(::GMM)` turns a full-covariance GMM into a diagonal-covariance GMM, by ignoring
off-diagonal elements"""
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

function Base.show(io::IO, h::History)
    println(io, Libc.strftime(h.t), ": ", h.s)
end

history(gmm::GaussianMixture) = gmm.hist

function Base.writemime(io::IO, ::MIME{symbol("text/plain")}, hist::Vector{History})
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
if VERSION < v"0.4.0-dev"
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
else
#    @doc """`convert(GMM{::Type}, GMM)` convert the GMM to a different floating point type""" ->
    function Base.convert{Td,Ts}(::Type{GMM{Td}}, gmm::GMM{Ts})
        Ts == Td && return gmm
        h = vcat(gmm.hist, History(string("Converted to ", Td)))
        w = map(Td, gmm.w)
        μ = map(Td, gmm.μ)
        gmmkind = kind(gmm)
        if gmmkind == :full
            Σ = map(eltype(FullCov{Td}),  gmm.Σ)
        elseif gmmkind == :diag
            Σ = map(Td, gmm.Σ)
        else
            error("Unknown kind")
        end
        GMM(w, μ, Σ, h, gmm.nx)
    end
    function Base.convert{Td,Ts}(::Type{VGMM{Td}}, vg::VGMM{Ts})
        Ts == Td && return vg
        h = vcat(vg.hist, History(string("Converted to ", Td)))
        W = map(eltype(FullCov{Td}), vg.W)
        π = convert(GMMprior{Td}, vg.π)
        VGMM(vg.n, vg.d, π, map(Td,vg.α), map(Td, vg.β), map(Td,vg.m),
             map(Td, vg.ν), W, h)
    end
    function Base.convert{Td,Ts}(::Type{GMMprior{Td}}, p::GMMprior{Ts})
        Ts == Td && return p
        GMMprior(map(Td, p.α₀), map(Td, p.β₀), map(Td, p.m₀), map(Td, p.ν₀), map(Td, p.W₀))
    end
end
