ccall(:jl_zero_subnormals, Bool, (Bool,), true)
using NumericExtensions
using BigData
using Distributions
using Clustering
using HDF5, JLD
using MAT

require("gmmtypes.jl")

include("gmms.jl")
include("train.jl")
include("io.jl")
include("stats.jl")
include("rand.jl")
include("recognizer.jl")
