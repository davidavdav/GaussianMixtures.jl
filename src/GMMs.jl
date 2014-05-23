## GMMs.jl  Some functions for a diagonal covariance Gaussian Mixture Model
## (c) 2013 David A. van Leeuwen

## This module also contains some rudimentary code for speaker
## recognition, perhaps this should move to another module.

module GMMs

include("nomodule.jl")

export GMM, CSstats, Stats, IExtractor, History, Data, DataOrMatrix, split, em!, map, llpg, post, history, show, stats, cstats, dotscore, savemat, readmat, nparams, means, covars, weights, setmem, vec, rand

end
