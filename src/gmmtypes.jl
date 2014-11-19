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

abstract GaussianMixture{T,CT}

## support for two kinds of covariance matrix
## Full covariance is represented by inverse cholesky of the covariance matrix, 
## i.e., Σ^-1 = ci * ci'
typealias DiagCov{T} Matrix{T}
typealias FullCov{T} Vector{Triangular{T,Matrix{T},:U,false}} 

## GMMs can be of type FLoat32 or Float64, and diagonal or full
type GMM{T<:FloatingPoint, CT<:Union(Matrix,Vector)} <: GaussianMixture{T,CT}
    n::Int                      # number of Gaussians
    d::Int                      # dimension of Gaussian
    w::Vector{T}                # weights: n
    μ::Matrix{T}                # means: n x d
    Σ::CT                       # covars n x d or n x d^2
    hist::Vector{History}       # history
    nx::Int                     # number of points used to train the GMM
    function GMM(w::Vector{T}, μ::Matrix{T}, Σ::Union(DiagCov{T},FullCov{T}), 
                 hist::Vector, nx::Int)
        n = length(w)
        isapprox(1, sum(w)) || error("weights do not sum to one")
        d = size(μ, 2)
        n == size(μ, 1) || error("Inconsistent number of means")
        if isa(Σ, Matrix)
            (n,d) == size(Σ) || error("Inconsistent covar dimension")
        else
            n == length(Σ) || error("Inconsistent number of covars")
            for (i,S) in enumerate(Σ) 
                (d,d) == size(S) || error(@sprintf("Inconsistent dimension for %d", i))
##                isposdef(S) || error(@sprintf("Covariance %d not positive definite", i))
            end
        end
        new(n, d, w, μ, Σ, hist, nx)
    end
end
GMM{T<:FloatingPoint}(w::Vector{T}, μ::Matrix{T}, Σ::Union(DiagCov{T},FullCov{T}), 
                      hist::Vector, nx::Int) = GMM{T, typeof(Σ)}(w, μ, Σ, hist, nx)

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
type Cstats{T<:FloatingPoint, CT<:Union(Matrix,Vector)}
    N::Vector{T}
    F::Matrix{T}
    S::CT
    function Cstats(n::Vector{T}, f::Matrix{T}, s::Union(Matrix{T},Vector{Matrix{T}}))
        size(n,1) == size(f,1) || error("Inconsistent size 0th and 1st order stats")
        if size(n) == size(s)   # full covariance stats
            all([size(f,2) == size(ss,1) == size(ss,2) for ss in s]) || error("inconsistent size 2st and 2nd order stats")           
       else 
            size(f) == size(s) || error("inconsistent size 1st and 2nd order stats")
        end
        new(n, f, s)
    end
end
Cstats{T<:FloatingPoint}(n::Vector{T}, f::Matrix{T}, s::Union(Matrix{T}, Vector{Matrix{T}})) = Cstats{T,typeof(s)}(n, f, s)
Cstats(t::Tuple) = Cstats(t...)

## A data handle, either in memory or on disk, perhaps even mmapped but I haven't seen any 
## advantage of that.  It contains a list of either files (where the data is stored)
## or data units.  The point is, that in processing, these units can naturally be processed
## independently.  

## The API is a dictionary of functions that help loading the data into memory
## Compulsory is: :load, useful is: :size
type Data{T,VT<:Union(Matrix,String)}
    list::Vector{VT}
    API::Dict{Symbol,Function}
    Data(list::Union(Vector{VT},Vector{Matrix{T}}), API::Dict{Symbol,Function})=new(list,API)
end
Data{T}(list::Vector{Matrix{T}}) = Data{T, eltype(list)}(list, Dict{Symbol,Function}())
Data{S<:String}(list::Vector{S}, t::DataType, API::Dict{Symbol,Function}) = Data{t, S}(list, API)

typealias DataOrMatrix{T} Union(Data{T}, Matrix{T})
