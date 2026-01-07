using FileIO
using JLD2
using GaussianMixtures
using Distributed
using DelimitedFiles
using RDatasets
using Test
using ScikitLearnBase
using Random

import Base.isapprox
isapprox(a::Tuple, b::Tuple) = all(Bool[isapprox(x, y) for (x, y) in zip(a, b)])

GM = GaussianMixtures  # alias

setmem(0.1)
include("data.jl")
include("bayes.jl")
include("train.jl")
include("scikitlearn.jl")
