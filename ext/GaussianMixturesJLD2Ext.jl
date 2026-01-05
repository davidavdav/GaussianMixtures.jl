module GaussianMixturesJLD2Ext

using GaussianMixtures
using JLD2
using FileIO

# From src/io.jl
function FileIO.save(filename::AbstractString, name::AbstractString, gmm::GMM)
    jldopen(filename, "w") do file
        # addrequire(file, GaussianMixtures)
        write(file, name, gmm)
    end
end

function FileIO.save(filename::AbstractString, name::AbstractString, gmms::Array{GMM})
    jldopen(filename, "w") do file
        # addrequire(file, GaussianMixtures)
        write(file, name, gmms)
    end
end

# From src/data.jl

## default load function
function _load(file::AbstractString)
    FileIO.load(file, "data")
end

## default size function
function _size(file::AbstractString)
    jldopen(file, "r") do fd
        size(fd["data"])
    end
end


## Data([strings], type; load=loadfunction, size=sizefunction)
function GaussianMixtures.Data(files::Vector{S}, datatype::DataType; kwargs...) where {S<:AbstractString}
    all([isa((k, v), (Symbol, Function)) for (k, v) in kwargs]) || error("Wrong type of argument")
    d = Dict{Symbol,Function}([kwargs...])
    if !haskey(d, :load)
        d[:load] = _load
        d[:size] = _size
    end
    Data(files, datatype, d)
end

## constructor for a plain file.
GaussianMixtures.Data(file::AbstractString, datatype::DataType, load::Function) = Data([file], datatype, load)
GaussianMixtures.Data(file::AbstractString, datatype::DataType; kwargs...) = Data([file], datatype; kwargs...)


end # module
