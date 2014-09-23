## GMMs.jl  Some functions for a diagonal covariance Gaussian Mixture Model
## (c) 2013 David A. van Leeuwen

## This module also contains some rudimentary code for speaker
## recognition, perhaps this should move to another module.

module GMMs

## some init code.  Turn off subnormal computation, as it is slow.  This is a global setting...
ccall(:jl_zero_subnormals, Bool, (Bool,), true)
using NumericExtensions
using BigData
using Distributions
using HDF5, JLD
using MAT

include("gmmtypes.jl")

include("gmms.jl")
include("io.jl")
include("stats.jl")
include("rand.jl")
include("recognizer.jl")

export GMM, CSstats, Stats, IExtractor, History, Data, DataOrMatrix, split, em!, map, llpg, posterior, history, show, stats, cstats, dotscore, savemat, readmat, nparams, means, covars, weights, setmem, vec, rand, ivector

end
