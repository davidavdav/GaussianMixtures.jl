## gmm/types.jl  The type that implements a GMM.
## (c) 2013--2015 David A. van Leeuwen

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

using Printf

"""
`History`, a type to record the history of how a GMM is built.
"""
struct History
    """timestamp"""
    t::Float64
    """description"""
    s::AbstractString
end
History(s::AbstractString) = History(time(), s)

"""
`GaussianMixture`, an abstract type for a mixture of full-covariance or diagonal-covariance Gaussian
distributions
"""
abstract type GaussianMixture{T,CT}; end

## support for two kinds of covariance matrix
## Full covariance is represented by inverse cholesky of the covariance matrix,
## i.e., Σ^-1 = ci * ci'
DiagCov{T} = AbstractArray{T,2}
FullCov{T} = Vector{UpperTriangular{T,Matrix{T}}}
CovType{T} = Union{DiagCov{T}, FullCov{T}}

VecOrMat{T} = Union{Vector{T},AbstractArray{T,2}}
MatOrVecMat{T} = Union{AbstractArray{T,2}, Vector{AbstractArray{T,2}}}

## GMMs can be of type FLoat32 or Float64, and diagonal or full
"""
`GMM` is the type that stores information of a Guassian Mixture Model.  Currently two main covariance
types are supported: full covarariance and diagonal covariance.
"""
mutable struct GMM{T<:AbstractFloat, CT<:CovType{T}} <: GaussianMixture{T,CT}
    "number of Gaussians"
    n::Int
    "dimension of Gaussian"
    d::Int
    "weights (size n)"
    w::Vector{T}
    "means (size n x d)"
    μ::Matrix{T}
    "covariances (size n x d for diagonal, or n x (d^2) for full)"
    Σ::CT
    "history"
    hist::Vector{History}
    "number of points used to train the GMM"
    nx::Int
    function GMM{T,CT}(w::Vector{T}, μ::AbstractArray{T,2}, Σ::CT,
                               hist::Vector, nx::Int) where{T, CT}
        n = length(w)
        isapprox(1, sum(w)) || error("weights do not sum to one")
        d = size(μ, 2)
        n == size(μ, 1) || error("Inconsistent number of means")
        if isa(Σ, Matrix)
            (n,d) == size(Σ) || error("Inconsistent covar dimension")
        else
            n == length(Σ) || error(@sprintf("Inconsistent number of covars %d != %d", n, length(Σ)))
            for (i,S) in enumerate(Σ)
                (d,d) == size(S) || error(@sprintf("Inconsistent dimension for %d", i))
##                isposdef(S) || error(@sprintf("Covariance %d not positive definite", i))
            end
        end
        new(n, d, w, μ, Σ, hist, nx)
    end
end
GMM(w::Vector{T}, μ::AbstractArray{T,2}, Σ::Union{DiagCov{T},FullCov{T}},
                      hist::Vector, nx::Int) where {T<:AbstractFloat} = GMM{T, typeof(Σ)}(w, μ, Σ, hist, nx)

## Variational Bayes GMM types.

## Please note our pedantic use of the Greek letter ν (nu), don't confuse this with Latin v!
## The index-0 "₀" is part of the identifier.
"""
`GMMprior` is a type that holds the prior for training GMMs using Variational Bayes.
"""
struct GMMprior{T<:AbstractFloat}
    "effective prior number of observations"
    α₀::T
    β₀::T
    "prior on the mean μ"
    m₀::Vector{T}
    "scale of precision Λ"
    ν₀::T
    "prior of the precision Λ"
    W₀::Matrix{T}
end

## In Variational Bayes, the GMM is not specified by point estimates of the paramters,
## but distributions over these parameters.
## These are Dirichlet for the weights and Gaussian-Wishart for the mean and precision.
## These distributions have parameters themselves, and these are stored in this type...
"""
`VGMM` is the type that is used to store a GMM in the Variational Bayes training.
"""
mutable struct VGMM{T} <: GaussianMixture{T,Any}
    "number of Gaussians"
    n::Int
    "dimension of Gaussian"
    d::Int
    "The prior used in this VGMM"
    π::GMMprior{T}
    "Dirichlet, size n"
    α::Vector{T}
    "scale of precision, size n"
    β::Vector{T}
    "means of means, size n * d"
    m::Matrix{T}
    "no. degrees of freedom, size n"
    ν::Vector{T}
    "scale matrix for precision? size n * (d * d)"
    W::FullCov{T}
    "history"
    hist::Vector{History}
end


## UBM-centered and scaled stats.
## This structure currently is useful for dotscoring, so we've limited the
## order to 1.  Maybe we can make this more general allowing for uninitialized second order
## stats?

## We store the stats in a (ng * d) structure, i.e., not as a super vector yet.
## Perhaps in ivector processing a supervector is easier.
"""
`CSstats` a type holding centered and scaled zeroth and first order GMM statistics
"""
struct CSstats{T<:AbstractFloat}
    "zeroth order stats"
    n::Vector{T}          # zero-order stats, ng
    "first order stats"
    f::Matrix{T}          # first-order stats, ng * d
    function CSstats{T}(n::Vector, f::Matrix) where{T}
        @assert size(n,1)==size(f, 1)
        new(n,f)
    end
end
CSstats(n::Vector{T}, f::Matrix{T}) where {T<:AbstractFloat} = CSstats{T}(n, f)
## special case for tuple (why would I need this?)
CSstats(t::Tuple) = CSstats(t[1], t[2])

## Cstats is a type of centered but un-scaled stats, necessary for i-vector extraction
"""
`Cstats`, a type holding centered zeroth, first and second order GMM statistics
"""
struct Cstats{T<:AbstractFloat, CT<:VecOrMat}
    "zeroth order stats"
    N::Vector{T}
    "first order stats"
    F::Matrix{T}
    "second order stats"
    S::CT
    function Cstats{T,CT}(n::Vector{T}, f::Matrix{T}, s::MatOrVecMat{T}) where{T, CT}
        size(n,1) == size(f,1) || error("Inconsistent size 0th and 1st order stats")
        if size(n) == size(s)   # full covariance stats
            all([size(f,2) == size(ss,1) == size(ss,2) for ss in s]) || error("inconsistent size 1st and 2nd order stats")
       else
            size(f) == size(s) || error("inconsistent size 1st and 2nd order stats")
        end
        new(n, f, s)
    end
end
Cstats(n::Vector{T}, f::Matrix{T}, s::MatOrVecMat{T}) where {T<:AbstractFloat} = Cstats{T,typeof(s)}(n, f, s)
Cstats(t::Tuple) = Cstats(t...)

## A data handle, either in memory or on disk, perhaps even mmapped but I haven't seen any
## advantage of that.  It contains a list of either files (where the data is stored)
## or data units.  The point is, that in processing, these units can naturally be processed
## independently.

## The API is a dictionary of functions that help loading the data into memory
## Compulsory is: :load, useful is: :size
"""
`Data` is a type for holding an array of feature vectors (i.e., matrices), or references to
files on disk.  The data is automatically loaded when needed, e.g., by indexing.
"""
struct Data{T,VT<:Union{Matrix,AbstractString}}
    list::Vector{VT}
    API::Dict{Symbol,Function}
    function Data{T,VT}(list::Union{Vector{VT},Vector{Matrix{T}}}, API::Dict{Symbol,Function}) where{T,VT}
        return new(list,API)
    end
end
Data(list::Vector{Matrix{T}}) where {T} = Data{T, eltype(list)}(list, Dict{Symbol,Function}())
Data(list::Vector{S}, t::DataType, API::Dict{Symbol,Function}) where {S<:AbstractString} = Data{t, S}(list, API)

DataOrMatrix{T} = Union{Data{T}, Matrix{T}}
