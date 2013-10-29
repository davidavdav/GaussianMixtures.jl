## mfcc.jl
## (c) 2013 David A. van Leeuwen

module Features

export mfcc, warp

using Rasta
using SignalProcessing
using Misc

## This function accepts a vector of sample values, below we will generalize to arrays, 
## i.e., multichannel data
## Recoded from rastamat's "melfcc.m" (c) Dan Ellis. 
## Defaults here are HTK parameters, this is contrary to melfcc
function mfcc(x::Vector, sr::Number=16000.0; wintime=0.025, steptime=0.01, numcep=13, lifterexp=-22, 
              sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=sr/2, 
              nbands=20, bwidth=1.0, dcttype=3, fbtype="htkmel", usecmp=false, modelorder=0)
##    println("filter")
    if (preemph!=0) 
        x |= Filter([1, -preemph])
    end
##    println("pspec")
    pspec = powspec(x, sr, wintime=wintime, steptime=steptime, dither=dither)
##    println("aspec")
    aspec = audspec(pspec, sr, nfilts=nbands, fbtype=fbtype, minfreq=minfreq, maxfreq=maxfreq, sumpower=sumpower, bwidth=bwidth)
    if usecmp
        #  PLP-like weighting/compression
        aspec = postaud(aspec, maxfreq, fbtype)
    end
##    println("cep")
    if modelorder>0
        if dcttype != 1
            ## error
        end
        # LPC analysis 
        lpcas = dolpc(aspectrum, modelorder)
        # cepstra
        cepstra = lpc2cep(lpcas, numcep)
    else
        cepstra = spec2cep(aspec, numcep, dcttype)
    end
##    println("lift")
    cepstra = lifter(cepstra, lifterexp)'
    return (cepstra,pspec')
end
mfcc(x::Array, sr::Number=16000.0...) = @parallel (tuple) for i=1:size(x)[2] mfcc(x[:,i], sr...) end

function warp(x::Array, w=399)
    l = nrow(x)
    wl = min(w, l)
    hw = iround((wl+1)/2)
    erfinvtab = sqrt(2)*erfinv([1:wl]/hw - 1)
    rank = zeros(Int, size(x))
    if l<w
        index = sortperm(x, 1)
        for j=1:ncol(x)
            rank[index[:,j],j] = [1:l]
        end
    else
        for i=1:l
            s=max(1,i-hw+1)
            e=s+w-1
            if (e>l) 
                d = e-l
                e -= d
                s -= d
            end
            rank[i,:] = 1+sum(broadcast(.>, x[i,:], x[s:e,:]), 1) # sum over columns
        end
    end
    return erfinvtab[rank]        
end

function sdc(x::Array, n::Int=7, d::Int=1, p::Int=3, k::Int=7)
    (nx, nfea) = size(x)
    @assert n <= nfea
    xx = vcat(delta(x[:,1:n], d), zeros(typeof(x[1]), (k-1)*p, n))
    y = zeros(typeof(x[1], nx, n*k))
    for (dt,offset) = zip(0:p:p*k-1,0:n:n*k-1)
        y[:,offset+(1:n)] = xx[dt+(1:nx),:]
    end
    return y
end

end
