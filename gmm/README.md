Gaussian Mixture Models
=======================

This julia type is more specific than Dahua Lin [MixtureModels](https://github.com/lindahua/MixtureModels.jl), in that it deals only with normal (multivariate) distributions (a.k.a Gaussians), but it does so more efficiently. 

At this moment, we have implemented only diagonal covariance GMMs.  

Vector dimenesions
------------------

Some remarks on the dimension.  There are three main indexing variables:
 - The gaussian index 
 - The data point
 - The feature dimension
Often data is stores in 2D arrays, and computations can be done efficiently in 
matrix multiplications.  For this it is nice to have the data in standard row,column order
however, we can't have these consistently over all three indexes. 

My approach is do have:
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
    Î¼::Array{Float64}		      # means: n x d
    Î£::Array{Float64}          # covars n x d
    hist::Array{History}        # history
end
```
