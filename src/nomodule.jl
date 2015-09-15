using Distributions
using PDMats
using Clustering
using JLD
using Compat

include("compat.jl")
if !isdefined(:GMM)
    include("gmmtypes.jl")
    include("bayestypes.jl")
end
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

