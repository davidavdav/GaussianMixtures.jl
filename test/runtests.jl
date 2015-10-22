using JLD
using GaussianMixtures

setmem(0.1)
include("data.jl")
include("bayes.jl")
include("train.jl")
