using Distributions
using PDMats
using Clustering
using JLD
using Compat

include("compat.jl")
require("gmmtypes.jl")
require("bayestypes.jl")

include("gmms.jl")
include("train.jl")
include("io.jl")
include("stats.jl")
include("rand.jl")
include("data.jl")
include("recognizer.jl")

include("bayes.jl") 

include("distributions.jl")

## some init code.  Turn off subnormal computation, as it is slow.  This is a global setting...
set_zero_subnormals(true)

