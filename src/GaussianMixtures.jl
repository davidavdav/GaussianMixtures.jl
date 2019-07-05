## GaussianMixtures.jl  Some functions for potentially large Gaussian Mixture Models
## (c) 2013--2015 David A. van Leeuwen

__precompile__()

module GaussianMixtures

using Distributions
using PDMats
using Clustering
using JLD2
using FileIO
using Compat
using Logging

# define a more informative info level 
const moreInfo = LogLevel(-1)   # Info is LogLevel( 0 )
# if you want to see more info, set up a logger (outside of this module) like

# using Logging
# more_logger = ConsoleLogger(stderr, LogLevel(-1))
# global_logger(more_logger)
# use package...

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
