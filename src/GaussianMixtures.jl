## GaussianMixtures.jl  Some functions for potentially large Gaussian Mixture Models
## (c) 2013--2014 David A. van Leeuwen

module GaussianMixtures

using NumericExtensions
using Distributions
using PDMats
using Clustering
using HDF5, JLD
using Compat

include("compat.jl")
include("gmmtypes.jl")
include("bayestypes.jl")

include("compat.jl")
include("gmms.jl")
include("train.jl")
include("bayes.jl")
include("io.jl")
include("stats.jl")
include("rand.jl")
include("data.jl")
include("recognizer.jl")

include("distributions.jl")

export GMM, VGMM, GMMprior, CSstats, Cstats, History, Data, DataOrMatrix, 
   split, em!, map, llpg, avll, gmmposterior, history, show, stats, nparams, means, covars, weights, setmem, vec, rand, kind, dmap

## some init code.  Turn off subnormal computation, as it is slow.  This is a global setting...
set_zero_subnormals(true)

end
