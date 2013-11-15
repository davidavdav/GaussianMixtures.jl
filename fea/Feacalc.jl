module Feacalc

## Feacalc.  Feature calculation as used for speaker and language recognition. 

export feacalc, sad

using Features
using SignalProcessing
using WAV
using Rasta
using Misc

## if test==true, compute features with parmeters as we use in speaker recogntion
function feacalc(wavfile::String; dynrange::Float64=30., defaults=:spkid_toolkit)
    (x, sr) = wavread(wavfile)
    sr = convert(Float64, sr)       # more reasonable sr
    data = {"nx" => nrow(x), "sr" => sr, "source" => wavfile} # save some metadata
    x = mean(x, 2)[:,1]             # averave multiple channels for now
    preemp = 0.97
    preemp ^= 16000. / sr

    ## basic features
    (m, pspec, meta) = mfcc(x, sr, defaults)
    data["nftot"] = nrow(m)
 
    ## integrate power
    deltaf = size(pspec,2) / (sr/2)
    minfreqi = iround(300deltaf)
    maxfreqi = iround(4000deltaf)
    power = 10log10(sum(pspec[:,minfreqi:maxfreqi], 2))
    
    maxpow = maximum(power)
    speech = find(power .> maxpow - dynrange)
    
    m = m[speech,:]
    m = warp(m)
    data["speech"] = convert(Vector{Uint32}, speech)
    data["nf"] = nrow(m)
    meta["dynrange"] = dynrange
    meta["warp"] = 399          # the default
    return(convert(Array{Float32},m), data, meta)
end

function sad(pspec::Array{Float64,2}, sr::Float64, method=:energy; dynrange::Float64=30.)
    deltaf = size(pspec,2) / (sr/2)
    minfreqi = iround(300deltaf)
    maxfreqi = iround(4000deltaf)
    power = 10log10(sum(pspec[:,minfreqi:maxfreqi], 2))
    maxpow = maximum(power)
    speech = find(power .> maxpow - dynrange)
end

## listen to SAD
function sad(wavfile::String, speechout::String, silout::String)
    (x, sr) = wavread(wavfile)
    sr = convert(Float64, sr)       # more reasonable sr
    x = mean(x, 2)[:,1]             # averave multiple channels for now
    (m, pspec, meta) = mfcc(x, sr; preemph=0)
    sp = sad(pspec, sr)
    sl = iround(meta["steptime"] * sr)
    xi = zeros(Bool, size(x))
    for (i = sp)
        xi[(i-1)*sl+(1:sl)] = true
    end
    y = x[find(xi)]
    wavwrite(y, sr, speechout)
    y = x[find(!xi)]
    wavwrite(y, sr, silout)
end

end
