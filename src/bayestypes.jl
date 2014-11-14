## experiments with variational Bayes

## Please note our pedantic use of the Greek letter ν, don't confuse this with v!
type GMMprior{T<:FloatingPoint}
    α0::T                       # effective prior number of observations
    β0::T
    m0::Vector{T}               # prior on μ
    W0::Matrix{T}               # prior precision
    ν0::T                       # scale precision
end    

## In Variational Bayes, the GMM is not specified by point estimates of the paramters, but distributions
## over these parameters.  
## These are Dirichlet for the weights and Gaussian-Wishart for the mean and precision.  
## These distributions have parameters themselves, and these are stored in the type...
type VGMM{T<:FloatingPoint} <: GaussianMixture{T}
    n::Int                      # number of Gaussians
    d::Int                      # dimension of Gaussian
    π::GMMprior                 # The prior used in this VGMM
    α::Vector{T}                # Dirichlet, n
    β::Vector{T}                # scale of precision, n
    m::Matrix{T}                # means of means, n * d
    ν::Vector{T}                # no. degrees of freedom, n
    W::FullCov{T}               # scale matrix for precision? n * d * d
    hist::Vector{History}       # history
end
