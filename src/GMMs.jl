## GMMs.jl  Some functions for a diagonal covariance Gaussian Mixture Model
## (c) 2013 David A. van Leeuwen

## This module also contains some rudimentary code for speaker
## recognition, perhaps this should move to another module.

module GMMs

include("gmmtypes.jl")
include("gmms.jl")
include("stats.jl")
include("recognizer.jl")

export GMM, Cstats, History, split, em!, map, llpg, post, history, show, stats, cstats, dotscore, savemat, readmat, nparams, means, covars, weights, setmem

end
