## io.jl  Some functions for reading/writing GMMs.
## (c) 2013--2014 David A. van Leeuwen

## This code is for exchange with our octave / matlab based system

## save a single GMM
function JLD2.save(filename::AbstractString, name::AbstractString, gmm::GMM)
    jldopen(filename, "w") do file
        # addrequire(file, GaussianMixtures)
        write(file, name, gmm)
    end
end
## save multiple GMMs
function JLD2.save(filename::AbstractString, name::AbstractString, gmms::Array{GMM})
    jldopen(filename, "w") do file
        # addrequire(file, GaussianMixtures)
        write(file, name, gmms)
    end
end
