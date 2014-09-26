## GMMs.jl  Some functions for potentially large Gaussian Mixture Models
## (c) 2013 David A. van Leeuwen

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

export GMM, CSstats, Stats, IExtractor, History, Data, DataOrMatrix, split, em!, map, llpg, posterior, history, show, stats, cstats, savemat, readmat, nparams, means, covars, weights, setmem, vec, rand, ivector

end
