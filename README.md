GaussianMixtures
========================
A Julia package for Gaussian Mixture Models (GMMs).
----------
[![Unit test](https://github.com/davidavdav/GaussianMixtures.jl/actions/workflows/test.yml/badge.svg)](https://github.com/davidavdav/GaussianMixtures.jl/actions/workflows/test.yml)

This package contains support for Gaussian Mixture Models.  Basic training, likelihood calculation, model adaptation, and i/o are implemented.

This Julia type is more specific than Dahua Lin's [MixtureModels](http://distributionsjl.readthedocs.org/en/latest/mixture.html), in that it deals only with normal (multivariate) distributions (a.k.a Gaussians), but it does so more efficiently, hopefully.  We have support for switching between GMM and MixtureModels types. 

At this moment, we have implemented both diagonal covariance and full covariance GMMs, and full covariance variational Bayes GMMs.  

In training the parameters of a GMM using the Expectation Maximization (EM) algorithm, the inner loop (computing the Baum-Welch statistics) can be executed efficiently using Julia's standard parallelization infrastructure, e.g., by using SGE.  We further support very large data (larger than will fit in the combined memory of the computing cluster) through [BigData](https://github.com/davidavdav/BigData.jl), which has now been incorporated in this package. 

Install
----
```julia
Pkg.add("GaussianMixtures")
```

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

Please note that the choice for "data points run down" is the opposite of the convention used in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), so if you convert `GMM`s to `MixtureModel`s in order to benefit from the methods provided for these distributions, you need to transpose the data.  

Type
----

A simplified version of the type definition for the Gaussian Mixture Model is
```julia
type GMM
    n::Int                         # number of Gaussians
    d::Int                         # dimension of Gaussian
    w::Vector                      # weights: n
    μ::Array                       # means: n x d
    Σ::Union(Array, Vector{Array}) # diagonal covariances n x d, or Vector n of d x d full covariances
    hist::Array{History}           # history of this GMM
end
```
Currently, the variable `Σ` is heterogeneous both in form and interpretation, depending on whether it represents full covariance or diagonal covariance matrices.  

 - full covariance: `Σ` is represented by a `Vector` of `chol(inv(Σ), :U)`
 - diagonal covariance: `Σ` is formed by vertically stacking row-vectors of `diag(Σ)`

Constructors
------------

```julia
GMM(weights::Vector, μ::Array,  Σ::Array, hist::Vector)
```
This is the basic outer constructor for GMM.  Here we have
 - `weights` a Vector of length `n` with the weights of the mixtures
 - `μ` a matrix of `n` by `d` means of the Gaussians
 - `Σ` either a matrix of `n` by `d` variances of diagonal Gaussians,
   or a vector of `n` `Triangular` matrices of `d` by `d`,
   representing the Cholesky decomposition of the full covariance
 - `hist` a vector of `History` objects describing how the GMM was
   obtained. (The type `History` simply contains a time `t` and a comment
   string `s`)

```julia
GMM(x::Matrix; kind=:diag)
GMM(x::Vector)
```
Create a GMM with 1 mixture, i.e., a multivariate Gaussian, and initialize with mean an variance of the data in `x`.  The data in `x` must be a `nx` x `d` Matrix, where `nx` is the number of data points, or a Vector of length `nx`. 

```julia
GMM(n:Int, x::Matrix; method=:kmeans, kind=:diag, nInit=50, nIter=10, nFinal=nIter)
```
Create a GMM with `n` mixtures, given the training data `x` and using the Expectation Maximization algorithm.  There are two ways of arriving at `n` Gaussians: `method=:kmeans` uses K-means clustering from the Clustering package to initialize with `n` centers.  `nInit` is the number of iterations for the K-means algorithm, `nIter` the number of iterations in EM.  The method `:split` works by initializing a single Gaussian with the data `x` and subsequently splitting the Gaussians followed by retraining using the EM algorithm until `n` Gaussians are obtained.  `n` must be a power of 2 for `method=:split`.  `nIter` is the number of iterations in the EM algorithm, and `nFinal` the number of iterations in the final step. 

```julia
GMM(n::Int, d::Int; kind=:diag)
```
Initialize a GMM with `n` multivariate Gaussians of dimension `d`.
The means are all set to **0** (the origin) and the variances to
**I**, which is silly by itself.  If `kind=:full` is specified, the
covariances are full rather than diagonal.  One should replace the
values of the weights, means and covariances afterwards.

Training functions
-------

```julia
em!(gmm::GMM, x::Matrix; nIter::Int = 10, varfloor=1e-3)
```
Update the parameters of the GMM using the Expectation Maximization (EM) algorithm `nIter` times, optimizing the log-likelihood given the data `x`.   The function `em!()` returns a vector of average log likelihoods for each of the intermediate iterations of the GMM given the training data.  

```julia
llpg(gmm::GMM, x::Matrix)
```
Returns `ll_ij = log p(x_i | gauss_j)`, the Log Likelihood Per Gaussian `j` given data point `x_i`.

```julia
avll(gmm::GMM, x::Matrix)
```
Computes the average log likelihood of the GMM given all data points,
further normalized by the feature dimension `d = size(x,2)`. A
1-mixture GMM then has an `avll` of `-(log(2π) +0.5 +log(σ))` if the
data `x` is distributed as a multivariate diagonal covariance Gaussian
with `Σ = σI`.  With `σ=1` we then have `avll≈-1.42`. 

```julia 
gmmposterior(gmm::GMM, x::Matrix)
```
Returns a tuple containing  `p_ij = p(j | gmm, x_i)`, the posterior probability that data point `x_i` 'belongs' to Gaussian `j`, and the Log Likelihood Per Gaussian `j` given data point `x_i` as in `llpg`. 

```julia
history(gmm::GMM)
```
Shows the history of the GMM, i.e., how it was initialized, split, how the parameters were trained, etc.  A history item contains a time of completion and an event string. 

You can examine a minimal example [using GMM for clustering](https://gist.github.com/mbeltagy/22e7fdf7e3be3fbfd97f1bde7a789b91). 

Other functions
---------------

 - `split(gmm; minweight=1e-5, covfactor=0.2)`: Doubles the number of Gaussians by splitting each Gaussian into two Gaussians.  `minweight` is used for pruning Gaussians with too little weight, these are replaced by an extra split of the Gaussian with the highest weight.  `covfactor` controls how far apart the means of the split Gaussian are positioned. 
 - `kind(gmm)`: returns `:diag` or `:full`, depending on the type of covariance matrix
 - `eltype(gmm)`: returns the datatype of `w`, `μ` and `Σ` in the GMM
 - `weights(gmm)`: returns the weights vector `w`
 - `means(gmm)`: returns the means `μ` as an `n` by `d` matrix
 - `covars(gmm)`: returns the covariances `Σ`
 - `copy(gmm)`: returns a deep copy of the GMM
 - `full(gmm)`: returns a full covariance GMM based on `gmm`
 - `diag(gmm)`: returns a diagonal GMM, by ignoring off-diagonal elementsin `gmm`

Converting the GMMs
-------------------
The element type in the GMM can be changed like you would expect.  We also have import and export functions for `MixtureModel`, which currently only has `Float64` element type. 
 - `convert(::Type{GMM{datatype}}, gmm)`: convert the data type of the GMM
 - `float16(gmm)`, `float32(gmm)`, `float64(gmm)`: convenience functions for `convert()`
 - `MixtureModel(gmm)`: construct an instance of type `MixtureModel` from the GMM.  Please note that for functions like `pdf(m::MixtureModel, x::Matrix)` the data `x` run "sideways" rather than "down" as in this package. 
 - `GMM(m::MixtureModel{Multivariate,Continuous,MvNormal})`: construct a GMM from the right kind of MixtureModel. 

Paralellization
---------------
Training a large GMM with huge quantities of data can take a significant amount of time.  We have built-in support for the parallelization infrastructure in Julia. 

The method `stats()`, which is at the heart of EM algorithm, can detect multiple processors available (through `nprocs()`).  If there is more than 1 processor available, the data is split into chunks, each chunk is mapped to a separate processor, and afterwards all the statistics from the sub-processes are aggregated.  In an SGE environment you can obtain more cores (in the example below 20) by issuing

```julia
using ClusterManagers
ClusterManagers.addprocs_sge(20)                                        
@everywhere using GaussianMixtures
```

The size of the GMM and the data determine whether or not it is advantageous to do this. 

Memory
------
The `stats()` method (see below) needs to be very efficient because for many algorithms it is at the inner loop of the calculation.  We have a highly optimized BLAS friendly and parallizable implementation, but this requires a fair bit of memory.  Therefore the input data is processed in blocks in such a way that only a limited amount of memory is used.  By default this is set at 2GB, but it can be specified though a gobal setting:

```julia
setmem(gig) 
```
Set the memory approximately used in `stats()`, in Gigabytes. 

Baum-Welch statistics
--------------------

At the heart of EM training, and to many other operations with GMMs, lies the computation of the Baum-Welch statistics of the data when aligning them to the GMM.  We have optimized implementations of the basic calculation of the statistics:

 - `stats(gmm::GMM, x::Matrix; order=2, parallel=true)` Computes the Baum-Welch statistics up to order `order` for the alignment of the data `x` to the Universal Background GMM `gmm`.  The 1st and 2nd order statistics are retuned as an `n` x `d` matrix, so for obtaining statistics in supervector format, flattening needs to be carried out in the right direction.  Theses statistics are _uncentered_. 

Random GMMs
-----------
Sometimes is it insteresting to generate random GMMs, and use these to genrate random points. 
```julia
g = rand(GMM, n, d; kind=:full, sep=2.0)
```
This generates a GMM with normally distributed means according to N(x|μ=0,Σ=sep*I).  The covariance matrices are also chosen random. 

```julia
rand(g::GMM, n)
```
Generate `n` datapoints sampled from the GMM, resulting in a `n` times `g.d` array. 

Speaker recognition methods
----------------------------

The following methods are used in speaker- and language recognition, they may eventually move to another module. 

```julia
csstats(gmm::GMM, x::Matrix, order=2)
```
Computes _centered_ and _scaled_ statistics.  These are similar as above, but centered w.r.t the UBM mean and scaled by the covariance.  

```julia
CSstats(x::GMM, x::Matrix)
```
This constructor return a `CSstats` object for centered stats of order 1.  The type is currently defined as:
```julia
type CSstats
    n::Vector           # zero-order stats, ng
    f::Matrix          # first-order stats, ng * d
end
```
The CSstats type can be used for MAP adaptation and a simple but elegant dotscoring speaker recognition system. 

```julia
dotscore(x::CSstats, y::CSstats, r::Float64=1.) 
```
Computes the dot-scoring approximation to the GMM/UBM log likelihood ratio for a GMM MAP adapted from the UBM (means only) using the data from `x` and a relevance factor of `r`, and test data from `y`. 

```julia
maxapost(gmm::GMM, x::Matrix, r=16.; means::Bool=true, weights::Bool=false, covars::Bool=false)
```
Perform Maximum A Posterior (MAP) adaptation of the UBM `gmm` to the data from `x` using relevance `r`.  `means`, `weights` and `covars` indicate which parts of the UBM need to be updated. 

Saving / loading a GMM
----------------------

Using package JLD, two methods allow saving a GMM or an array of GMMs to disk:

```julia
using JLD
save(filename::String, name::String, gmm::GMM)
save(filename::String, name::String, gmms::Array{GMM})
```
This saves a GMM of an array of GMMs under the name `name` in a file `filename`. The data can be loaded back into a julia session using plain JLD's 

```julia
gmm = load(filename, name)
```

In case of using `kind=:full` covariance matrices make sure you have also loaded `LinearAlgebra` module into the current session. Without it JLD is unable to reconstruct `LinearAlgebra.UpperTriangular` type and gmm object won't be created
```julia
using JLD
using GaussianMixtures
using LinearAlgebra
gmm = load(filename, name)
```

Support for large amounts of training data
------
In many of the functions defined above, a `Data` type is accepted in the place where the data matrix `x` is indicated.  An object of type `Data` is basically a list of either matrices of filenames, see  [BigData](https://github.com/davidavdav/BigData.jl).

If `kind(x::Data)==:file`, then the matrix `x` is represented by vertically stacking the matrices that can be obtained by loading the files listed in `d` from disc.  The functions in GaussianMixtures try to run computations in parallel by processing the files in `d` simultaneously on multiple cores/machines, and they further try to limit the number of times the data needs to be loaded form disc.  In parallel execution on a computer cluster, an attempt is made to ensure the same data is always processed by the same worker, so that local file caching could work to your advantage. 

Variational Bayes training
-------------------
We have started support for Variational Bayes GMMs.  Here, the parameters of the GMM are not point estimates but are rather represented by a distribution.  Training of the parameters that govern these distributions can be carried out by an EM-like algorithm.

In our implementation, we follow the approach from section 10.2 of [Christopher Bishop's book](http://research.microsoft.com/en-us/um/people/cmbishop/PRML/).  In the current version of GaussianMixtures we have not attempted to optimize the code for efficiency.

The variational Bayes training uses two new types, a prior and a variational GMM:

```julia
type GMMprior
    α0                       # effective prior number of observations
    β0
    m0::Vector               # prior on μ
    ν0                       # scale precision
    W0::Matrix               # prior precision
end

type VGMM
    n::Int                   # number of Gaussians
    d::Int                   # dimension of Gaussian
    π::GMMprior              # The prior used in this VGMM
    α::Vector                # Dirichlet, n
    β::Vector                # scale of precision, n
    m::Matrix                # means of means, n * d
    ν::Vector                # no. degrees of freedom, n
    W::Vector{Matrix}        # scale matrix for precision? n * d * d
    hist::Vector{History}    # history
end
```
Please note that we currently only have full covariance VGMMs.  
Training using observed data `x` needs some initial GMM and a prior.

```julia
g = GMM(8, x, kind=:full, nIter=0) ## only do k-means initialization of GMM g
p = GMMprior(g.d, 0.1, 1.0)  ## set α0=0.1 and β0=1, and other values to a default
v = VGMM(g, p) ## initialize variational GMM v with g
```

Training can then proceed with
```julia
em!(v, x)
```

This EM training checks if the occupancy of the Gaussians still is nonzero after each M-step.  In case it isn't, the Gaussian is removed.  The effect is that the total number of Gaussians can reduce in this procedure.

Working with Distributions.jl 
-------------------

A GMM model can used to build a `MixtureModel` in the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package. For example:
```julia 
using GaussianMixtures
using Distributions
g = rand(GMM, 3, 4)
m = MixtureModel(g)
```
This can be conveniently use for sampling from the GMM, e.g.
```julia
sample= rand(m)
```
Furthermore, a Gaussian mixture model constructed using `MixtureModel` can be
converted to GMM via a constructor call 
```julia
gg = GMM(m)
`
