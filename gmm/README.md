Gaussian Mixture Models (GMMs)
=======================

This julia type is more specific than Dahua Lin's [MixtureModels](https://github.com/lindahua/MixtureModels.jl), in that it deals only with normal (multivariate) distributions (a.k.a Gaussians), but it does so more efficiently. 

At this moment, we have implemented only diagonal covariance GMMs.  

Vector dimensions
------------------

Some remarks on the dimension.  There are three main indexing variables:
 - The gaussian index 
 - The data point
 - The feature dimension

Often data is stored in 2D slices, and computations can be done efficiently as 
matrix multiplications.  For this it is nice to have the data in standard row,column order
however, we can't have these consistently over all three indices. 

My approach is to have:
 - The data index (`i`) always be a row-index
 - The feature dimenssion index (`k`) always to be a column index
 - The Gaussian index (`j`) to be mixed, depending on how it is used

Type
----

```julia
type GMM
    n::Int                      # number of Gaussians
    d::Int                      # dimension of Gaussian
    kind::Symbol                # :diag or :full---we'll take 'diag' for now
    w::Vector{Float64}          # weights: n
    μ::Array{Float64}           # means: n x d
    Σ::Array{Float64}           # covars n x d
    hist::Array{History}        # history
end
```

Constructors
------------

```julia
GMM(n::Int, d::Int)
```
Initialize a diagonal covariance GMM with `n` multivariate Gaussians of dimension `d`.  The means are all set to **0** (the origin) and the variances to 1. 

```julia
GMM(x::Array)
```
Create a GMM with 1 mixture, i.e., a multivaviate Gaussian, and initialize with mean an variance of the data in `x`.  The data in `x` must be a `nx` x `d` data array, where `nx` is the number of data points. 

```julia
GMM(x::Array, n::Int; nIter=10, nFinal=nIter)
```
Create a GMM with `n` mixtures (diagonal covariance multivariate Gaussians), by initializing with the data `x` and subsequently splitting the Gaussians and retaining using the EM algorithm until `n` Gaussians are obtained.  `n` must be a power of 2.  `nIter` is the number of iterations in the EM algorithm, and `nFinal` the number of iterations in the final step. 

```julia
split(gmm::GMM; minweight=1e-5, covfactor=0.2)
```
Double the number of Gaussians by splitting each Gaussian into two Gaussians.  `minweight` is used for pruning Gaussians with too little weight, these are replaced by an extra split of the Gaussian with the highest weight.  `covfactor` controls how far apart the means of the split Gaussian are positioned. 

```julia
em!(gmm::GMM, x::Array; nIter::Int = 10, varfloor::Float64=1e-3, logll=true)
```
Update the parameters of the GMM using the Expectation Maximization (EM) algoritm `nIter` times, optimizing the log-likelihood given the data `x`.  

```julia
llpg(gmm::GMM, x::Array)
```
Returns ll\_ij = log p(x\_i | gauss\_j), the log likelihood of Gaussian j given data point i.

```julia
avll(gmm::GMM, x)
```
Computes the averave log likelihood of the GMM given all data points, normalized by the feature dimension `d = size(x,2)`. A 1-mixture GMM has an `avll` of -σ if the data `x` is distributed as a multivariate diagonal covariance Gaussian with Σ = σI.  

```julia 
post(gmm::GMM, x::Array)
```
Returns p\_ij = p(j | gmm, x\_i), the posterior probability that data point `x_i` 'belongs' to Gaussian `j`.  

```julia
history(gmm::GMM)
```
Shows the history of the GMM, i.e., how it was initialized, split, how the paramteres were trained, etc.  A history item contains a time of completion and an event string. 

Speaker recognition methods
----------------------------

The following methods are used in speaker- and language recogntion, they may eventually move to another module. 

```julia
stats(gmm::GMM, x::Array, order=2)
```
Computes the Baum-Welch statistics up to order `order` for the alignment of the data `x` to the Universal Background GMM `gmm`.  The 1st and 2nd order statistics are retuned as an `n` x `d` matrix, so for obtaining a supervector flattening needs to be carried out in the rigt direction.  Theses statistics are _uncentered_. 

```julia
cstats(gmm::GMM, x::Array, order=2)
```
Computes _centered_ statistics.  These are similar as above, but centered w.r.t the UBM mean and normalized by the covariance.  

```julia
Cstats(x::GMM, x::Array)
```
This constructor return a `Cstats` object for centered stats of order 1.  The type is currently defined as:
```julia
type Cstats
    n::Vector{Float64}           # zero-order stats, ng
    f::Array{Float64,2}          # first-order stats, ng * d
end
```
The Cstats type can be used for i-vector extraction (not implemented yet), MAP adaptation and a simple but elegant dotscoring speaker recognition system. 

```julia
dotscore(x::Cstats, y::Cstats, r::Float64=1.) 
```
Computes the dot-scoring apporiximation to the GMM/UBM log likelihood ratio for a GMM MAP adapted from the UBM (means only) using the data from `x` and a relevance factor of `r`, and test data from `y`. 

```julia
map(gmm::GMM, x::Array, r::Float64=16.; means::Bool=true, weights::Bool=false, covars::Bool=false)
```
Perform Maximum A Posterior (MAP) adaptation of the UBM `gmm` to the data from `x` using relevance `r`.  `means`, `weights` and `covars` indicate which parts of the UBM need to be updated. 

Saving / loading a GMM
----------------------

We have some temporary methods to save/retrieve a GMM in octave (and also Matab, which is a trademark and not very much liked by us) compatible format, with names resembling those in the good-old `netlib' implementation from Nabney and Bishop. 

At some point we might move towards a more general HDF5 format, perhaps JLD. 

```julia
savemat(file::String, gmm::GMM) 
```
Saves the GMM in file `file`. 

```julia
readmat{T}(file, ::Type{T})
```
When called as `readmat(file, GMM)`, opens the file `file` and reads the gmm. 
