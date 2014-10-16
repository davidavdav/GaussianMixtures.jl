Gaussian Mixture Models (GMMs)
=======================

This package contains support for Gaussian Mixture Models.  Basic training, likelihood calculation, model adaptation, and i/o are implemented.

This Julia type is more specific than Dahua Lin's [MixtureModels](https://github.com/lindahua/MixtureModels.jl), in that it deals only with normal (multivariate) distributions (a.k.a Gaussians), but it does so more efficiently, hopefully. 

At this moment, we have implemented both diagonal covariance and full covariance GMMs. 

In training the parameters of a GMM using the Expectation Maximization (EM) algorithm, the inner loop (computing the Baum-Welch statistics) can be executed efficiently using Julia's standard parallelization infrastructure, e.g., by using SGE.  We further support very large data (larger than will fit in the combined memory of the computing cluster) though [BigData](https://github.com/davidavdav/BigData.jl). 

Vector dimensions
------------------

Some remarks on the dimension.  There are three main indexing variables:
 - The Gaussian index 
 - The data point
 - The feature dimension (for full covariance this adds to two dimensions)

Often data is stored in 2D slices, and computations can be done efficiently as 
matrix multiplications.  For this it is nice to have the data in standard row,column order. 
However, we can't have these consistently over all three indexes. 

My approach is to have:
 - The data index (`i`) always be a the first (row) index
 - The feature dimension index (`k`) always to be a the second (column) index
 - The Gaussian index (`j`) to be mixed, depending on how it is combined with either dimension above. 

The consequence is that "data points run down" in a matrix, just like records do in a DataFrame.  Hence, statistics per feature dimension occur consecutive in memory which may be advantageous for caching efficiency.  On the other hand, features belonging to the same data point are separated in memory, which probably is not according to the way they are generated, and does not extend to streamlined implementation.  The choice in which direction the data must run is an almost philosophical problem that I haven't come to a final conclusion about.  

Type
----

```julia
type GMM
    n::Int                         # number of Gaussians
    d::Int                         # dimension of Gaussian
    kind::Symbol                   # :diag or :full
    w::Vector                      # weights: n
    μ::Array                       # means: n x d
    Σ::Union(Array, Vector{Array}) # diagonal covariances n x d, or Vector n of d x d full covariances
    hist::Array{History}           # history of this GMM
end
```

Constructors
------------

```julia
GMM(n::Int, d::Int)
GMM(n::Int, d::Int; kind=:diag)
```
Initialize a GMM with `n` multivariate Gaussians of dimension `d`.  The means are all set to **0** (the origin) and the variances to **I**.  If `diag=:full` is specified, the covariances are full rather than diagonal. 

```julia
GMM(x::Matrix; kind=:diag)
GMM(x::Vector)
```
Create a GMM with 1 mixture, i.e., a multivariate Gaussian, and initialize with mean an variance of the data in `x`.  The data in `x` must be a `nx` x `d` Matrix, where `nx` is the number of data points, or a Vector of length `nx`. 

```julia
GMM(x::Matrix, n::Int, method=:kmeans; kind=:diag, nInit=50, nIter=10, nFinal=nIter)
```
Create a GMM with `n` mixtures, given the training data `x` and using the Expectation Maximization algorithm.  There are two ways of arriving at `n` Gaussians: `method=:kmeans` uses K-means clustering from the Clustering package to initialize with `n` centers.  `nInit` is the number of iterations for the K-means algorithm, `nIter` the number of iterations in EM.  The method `:split` works by initializing a single Gaussian with the data `x` and subsequently splitting the Gaussians followed by retraining using the EM algorithm until `n` Gaussians are obtained.  `n` must be a power of 2 for `method=:split`.  `nIter` is the number of iterations in the EM algorithm, and `nFinal` the number of iterations in the final step. 

```julia
split(gmm::GMM; minweight=1e-5, covfactor=0.2)
```
Double the number of Gaussians by splitting each Gaussian into two Gaussians.  `minweight` is used for pruning Gaussians with too little weight, these are replaced by an extra split of the Gaussian with the highest weight.  `covfactor` controls how far apart the means of the split Gaussian are positioned. 

```julia
em!(gmm::GMM, x::Matrix; nIter::Int = 10, varfloor=1e-3)
```
Update the parameters of the GMM using the Expectation Maximization (EM) algorithm `nIter` times, optimizing the log-likelihood given the data `x`.   The function `em!()` returns a vector of average log likelihoods for each of the intermediate iterations of the GMM given the training data.  

```julia
llpg(gmm::GMM, x::Matrix)
```
Returns `ll_ij = log p(x_i | gauss_j)`, the Log Likelihood Per Gaussian `j` given data point `i`.

```julia
avll(gmm::GMM, x::Matrix)
```
Computes the average log likelihood of the GMM given all data points, further normalized by the feature dimension `d = size(x,2)`. A 1-mixture GMM has an `avll` of `-σ` if the data `x` is distributed as a multivariate diagonal covariance Gaussian with `Σ = σI`.  

```julia 
posterior(gmm::GMM, x::Array)
```
Returns `p_ij = p(j | gmm, x_i)`, the posterior probability that data point `x_i` 'belongs' to Gaussian `j`.  

```julia
history(gmm::GMM)
```
Shows the history of the GMM, i.e., how it was initialized, split, how the parameters were trained, etc.  A history item contains a time of completion and an event string. 

Paralellization
---------------

The method `stats()`, which is at the heart of EM, can detect multiple processors available (through `nprocs()`).  If there is more than 1 processor available, the data is split into chunks, each chunk is mapped to a separate processor, and afterwards an aggregating operation collects all the statistics from the sub-processes.  In an SGE environment you can obtain more cores (in the example below 20) by issuing

```julia
using ClusterManagers
ClusterManagers.addprocs_sge(20)                                        
@everywhere using GMMs                                                  
```

Memory
------
The `stats()` method (see below) needs to be very efficient because for many algorithms it is at the inner loop of the calculation.  We have a highly optimized BLAS friendly and parallizable implementation, but this requires a fair bit of memory.  Therefore the input data is processed in blocks in sushc a way that only a limited amount of memory is used.  By default this is set at 2GB, but it can be specified though a gobal setting:

```julia
setmem(gig) 
```
Set the memory approximately used in `stats()`, in Gigabytes. 

Random GMMs
-----------
Sometimes is it insteresting to generate random GMMs, and use these to genrate random points. 
```julia
g = rand(GMM, n, d; kind=:full, sep=2.0)
```
This generates a GMM with normally distributed means according to N(x|μ=sep,Σ=I).  The covariance matrices are also chosen random. 

```julia
rand(g::GMM, n)
```
Generate `n` datapoints sampled from the GMM, resulting in a `n` times `g.d` array. 

Speaker recognition methods
----------------------------

The following methods are used in speaker- and language recognition, they may eventually move to another module. 

```julia
stats(gmm::GMM, x::Matrix, order=2; parallel=true, llhpf=false)
```
Computes the Baum-Welch statistics up to order `order` for the alignment of the data `x` to the Universal Background GMM `gmm`.  The 1st and 2nd order statistics are retuned as an `n` x `d` matrix, so for obtaining a supervector flattening needs to be carried out in the right direction.  Theses statistics are _uncentered_. 

```julia
csstats(gmm::GMM, x::Array, order=2)
```
Computes _centered_ and _scaled_ statistics.  These are similar as above, but centered w.r.t the UBM mean and scaled by the covariance.  

```julia
CSstats(x::GMM, x::Array)
```
This constructor return a `CSstats` object for centered stats of order 1.  The type is currently defined as:
```julia
type CSstats
    n::Vector{Float64}           # zero-order stats, ng
    f::Array{Float64,2}          # first-order stats, ng * d
end
```
The CSstats type can be used for MAP adaptation and a simple but elegant dotscoring speaker recognition system. 

```julia
dotscore(x::CSstats, y::CSstats, r::Float64=1.) 
```
Computes the dot-scoring approximation to the GMM/UBM log likelihood ratio for a GMM MAP adapted from the UBM (means only) using the data from `x` and a relevance factor of `r`, and test data from `y`. 

```julia
map(gmm::GMM, x::Matrix, r=16.; means::Bool=true, weights::Bool=false, covars::Bool=false)
```
Perform Maximum A Posterior (MAP) adaptation of the UBM `gmm` to the data from `x` using relevance `r`.  `means`, `weights` and `covars` indicate which parts of the UBM need to be updated. 

Saving / loading a GMM
----------------------

Using package JLD, two methods allow saving a GMM or an array of GMMs to disk:

```julia
save(filename::String, name::String, gmm::GMM)
save(filename::String, name::String, gmms::Array{GMM})
```
This saves a GMM of an array of GMMs under the name `name`  in a file `filename`. The data can be loaded back into a julia session using plain JLD's 

```julia
gmm = load(filename)[name]
```

