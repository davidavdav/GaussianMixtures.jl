## io.jl  Some functions for reading/writing GMMs. 
## (c) 2013--2014 David A. van Leeuwen

## This code is for exchange with our octave / matlab based system

## save a single GMM
function JLD.save(filename::String, name::String, gmm::GMM)
    jldopen(filename, "w") do file
        addrequire(file, "GaussianMixtures")
        write(file, name, gmm)
    end
end
## save multiple GMMs
function JLD.save(filename::String, name::String, gmms::Array{GMM})
    jldopen(filename, "w") do file
        addrequire(file, "GaussianMixtures")
        write(file, name, gmms)
    end
end

