## gmm/types.jl  The type that implements a GMM. 
## (c) 2013--2014 David A. van Leeuwen

## Some remarks on the dimension.  There are three main indexing variables:
## - The gaussian index 
## - The data point
## - The feature dimension
## Often data is stores in 2D arrays, and computations can be done efficiently in 
## matrix multiplications.  For this it is nice to have the data in standard row,column order
## however, we can't have these consistently over all three indexes. 

## My approach is do have:
## - The data index (i) always be a row-index
## - The feature dimenssion index (k) always to be a column index
## - The Gaussian index (j) to be mixed, depending on how it is used
## For full covar Σ, we need 2 feature dims.  For individuals covars to be
## consecutive in memory, the gaussian index should be _last_. 

## force Emacs utf-8: αβγδεζηθικλμνξοπρστυφχψω 

type History
    t::Float64
    s::String
end
History(s::String) = History(time(), s)

## typealias MatrixOrArray{T} Union(Matrix{T}, Vector{Matrix{T}})

## GMMs can be of type FLoat32 or Float64
type GMM{T<:FloatingPoint}
    n::Int                      # number of Gaussians
    d::Int                      # dimension of Gaussian
    kind::Symbol                # :diag or :full---we'll take 'diag' for now
    w::Vector{T}                # weights: n
    μ::Matrix{T}                 # means: n x d
    Σ::Union(Matrix{T},Vector{Matrix{T}}) # covars n x d
    hist::Vector{History}        # history
    function GMM(kind::Symbol, w::Vector, μ::Matrix, Σ::Array, hist::Vector)
        n = length(w)
        isapprox(1, sum(w)) || error("weights do not sum to one")
        d = size(μ, 2)
        n == size(μ, 1) || error("Inconsistent number of means")
        if kind == :diag
            (n,d) == size(Σ) || error("Inconsistent covar dimension")
        elseif kind == :full
            n == length(Σ) || error("Inconsistent number of covars")
            for (i,S) in enumerate(Σ) 
                (d,d) == size(S) || error(@sprintf("Inconsistent dimension for %d", i))
                isposdef(S) || error(@sprintf("Covariance %d not positive definite", i))
            end
        else
            error("Unknown kind")
        end
        new(n, d, kind, w, μ, Σ, hist)
    end
end
GMM{T<:FloatingPoint}(kind::Symbol, w::Vector{T}, μ::Matrix{T}, Σ::Union(Matrix{T},Vector{Matrix{T}}), hist::Vector) = GMM{T}(kind, w, μ, Σ, hist)

## UBM-centered and scaled stats.
## This structure currently is useful for dotscoring, so we've limited the
## order to 1.  Maybe we can make this more general allowing for uninitialized second order
## stats?

## We store the stats in a (ng * d) structure, i.e., not as a super vector yet.  
## Perhaps in ivector processing a supervector is easier. 
type CSstats{T<:FloatingPoint}
    n::Vector{T}          # zero-order stats, ng
    f::Matrix{T}          # first-order stats, ng * d
    function CSstats(n::Vector, f::Matrix)
        @assert size(n,1)==size(f, 1)
        new(n,f)
    end
end
CSstats{T<:FloatingPoint}(n::Vector{T}, f::Matrix{T}) = CSstats{T}(n, f)
## special case for tuple (why would I need this?)
CSstats(t::Tuple) = CSstats(t[1], t[2])

## Cstats is a type of centered but un-scaled stats, necessary for i-vector extraction
type Cstats{T<:FloatingPoint}
    N::Vector{T}
    F::Matrix{T}
    S::Union(Matrix{T}, Vector{Matrix{T}})
    function Cstats(n::Vector, f::Matrix, s::Array)
        @assert size(n,1) == size(f,1)
        if size(n) == size(s)   # full covariance stats
            @assert all([size(f,2) == size(ss,1) == size(ss,2) for ss in s])
       else 
            @assert size(f) == size(s)
        end
        new(n, f, s)
    end
end
Cstats{T<:FloatingPoint}(n::Vector{T}, f::Matrix{T}, s::Union(Matrix{T}, Vector{Matrix{T}})) = Cstats{T}(n, f, s)
Cstats(t::Tuple) = Cstats(t...)
