## experiments with variational Bayes

type GMMprior{T<:FloatingPoint}
    α0::T                     # effective prior number of observations
    β0::T
    m0::Vector{T}               # prior on μ
    W0::Matrix{T}               # prior precision
    ν0::T                       # idem
end    

## In Variational Bayes, the GMM is not specified by point estimates of the paramters, but distributions
## over these parameters.  
## These are Dirichlet for the weights and Gaussian-Wishart for the mean and precision.  
## These distributions have parameters themselves, and these are stored in the type...
type VGMM{T<:FloatingPoint}
    n::Int                      # number of Gaussians
    d::Int                      # dimension of Gaussian
    kind::Symbol                # :diag or :full---we'll take 'diag' for now
    α::Vector{T}                # Dirichlet, n
    β::Vector{T}                # scale of precision, n
    m::Matrix{T}                # means of means, n * d
    nu::Vector{T}               # no. degrees of freedom, n
    W::Vector{Matrix{T}}        # scale matrix for precision? n * d * d
    hist::Vector{History}       # history
end
