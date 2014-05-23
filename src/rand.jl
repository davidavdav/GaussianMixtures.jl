## can we generate random GMMs and sample from them?

using Distributions

## This function initializes a random GMM, with random means and random covariances
## 
function Base.rand(::Type{GMM}, ng::Int, d::Int; sep=2.0, kind=:full)
    μ = sep * randn(ng, d)
    if kind==:diag
        Σ = rand(Chisq(3), ng, d)
    else
        Σ = Array(Float64, d, d, ng)
        for i=1:ng
            T = randn(d,d)
            Σ[:,:,i] = T' * T
        end
    end
    w = ones(ng)/ng
    hist = History(@sprintf("Generated random %s covariance GMM with %d Gaussians of dimension %d",
                            kind, ng, d))
    GMM(kind, w, μ, Σ, [hist])
end

## local helper
function binsearch{T}(x::T, a::Vector{T})
    issorted(a) || error("Array needs to be sorted")
    mi = 1
    ma = length(a)
    if x < a[mi]
        return 0
    elseif x >= a[ma]
        return ma
    end
    while ma - mi > 1
        h = mi + div(ma-mi,2)
        if x > a[h]
            mi = h
        else
            ma = h
        end
    end
    return mi
end


## This function samples n data points from a GMM.  This is pretty slow. 
function Base.rand(gmm::GMM, n::Int)
    x = Array(Float64, n, gmm.d)
    cw = cumsum(gmm.w)
    for i=1:n
        ind = binsearch(rand(), cw)+1
        if gmm.kind == :diag
            x[i,:] = gmm.μ[ind,:] + sqrt(gmm.Σ[ind,:]) .* randn(gmm.d)'
        else
            x[i,:] = rand(MvNormal(vec(gmm.μ[ind,:]), gmm.Σ[:,:,ind]))
        end
    end
    x
end
