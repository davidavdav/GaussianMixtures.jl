## GaussianMixtures.jl  Some functions for potentially large Gaussian Mixture Models
## (c) 2013--2015 David A. van Leeuwen

__precompile__()

module GaussianMixtures

using Distributions
using PDMats
using Clustering
using JLD
using Compat

include("compat.jl")
include("gmmtypes.jl")
include("bayestypes.jl")

include("gmms.jl")
include("train.jl")
include("bayes.jl")
include("io.jl")
include("stats.jl")
include("rand.jl")
include("data.jl")
include("recognizer.jl")

include("distributions.jl")
include("scikitlearn.jl")

export GMM, VGMM, GMMprior, CSstats, Cstats, History, Data, DataOrMatrix,
gmmsplit, em!, maxapost, llpg, avll, gmmposterior, sanitycheck!,
history, show, stats, nparams, means, covars, weights, setmem, vec, rand, kind, dmap

## some init code.  Turn off subnormal computation, as it is slow.  This is a global setting...
set_zero_subnormals(true)

end
