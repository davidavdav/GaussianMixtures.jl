## Some Signal Processing routines for julia
## (c) 2013 David van Leeuwen
## Recoded from Octave's signal processing implementations, by Paul Kienzle, jwe && jh

## These routines are styled after their matlab (we don't like matlab, which is a trademark)
## counterparts.  

module SignalProcessing

export hamming, hanning, specgram, levinson, toeplitz, Filter, filter

include("filtertype.jl")
include("filter.jl")

## Freely after Octave's hamming by AW <Andreas.Weingessel@ci.tuwien.ac.at>
function hamming(n::Int) 
    if n==1
        return 1;
    end
    n -= 1
    return 0.54 - 0.46 * cos(2π*[0:n]/n)
end
    
function hanning(n::Int) 
    if (n==1) return 1 end
    n -= 1
    return 0.5 - 0.5*cos(2π*[0:n]/n)
end

## Freelyly after Paul Kienzle <pkienzle@users.sf.net>
## Note: in Octave, the default is a hanning window (implemented here)
## In Matlab, and we can't stress enough that we don't like Matlab, the default is hamming. 
## So it is better to specify the window!

## The spectogram is returned as nfreq * nframes array, so time is running right
## This is different from the conventions we use in Features
function specgram{T}(x::Vector{T}, n::Int=256; sr::Real=8000., window=hamming(n), overlap::Int=n/2)
    if typeof(window) == Int
        window = hanning(n)     # this is sort-of odd
    end
#    println(length(window), " ", n)
    winsize = min(length(window),n)
    step = winsize - overlap

    offset = 1:step:length(x)-winsize # truncate frames to integer amount
    lo = length(offset)
    S = zeros(n, lo)
    for (i,j)=zip(1:lo, offset)
        S[1:winsize,i] = x[j:j+winsize-1] .* window
    end
    S = fft(S,1)
    nn = iceil(n/2)
    S = S[1:nn,:]
    
    f = [0:nn-1] * sr/n
    t = offset / sr

    return (S, f, t)
end

## Freely after octave's implementation, by Paul Kienzle <pkienzle@users.sf.net>
## untested
## only returns a, v
function levinson{T<:Real}(acf::Vector{T}, p::Int)
    if length(acf)<1
        ## error
    end
    if p<0 
        ## error
    end
    if p<100
        ## direct solution [O(p^3), but no loops so slightly faster for small p]
        ## Kay & Marple Eqn (2.39)
        R = toeplitz(acf[1:p], conj(acf[1:p]))
        a = R \ -acf[2:p+1]
        unshift!(a, 1)
        v = real(a.*conj(acf[1:p+1]))
    else
        ## durbin-levinson [O(p^2), so significantly faster for large p]
        ## Kay & Marple Eqns (2.42-2.46)
        ref = zeros(p)
        g = -acf[2]/acf[1]
        a = [g]
        v = real((1-abs2(g)) * acf[1])
        ref[1] = g
        for t=2:p
            g = -(acf[t+1] + dot(a,acf[t:-1:2])) / v
            a = [a + g*conj(a[t-1:-1:1]), g]
            v *= 1 - abs2(g)
            ref[t] = g
        end
        unshift!(a, 1)
    end
    return (a,v)
end

function levinson{T<:Real}(acf::Array{T}, p::Int) 
    (nr,nc) = size(acf)
    a = zeros(p+1, nc)
    v = zeros(p+1, nc)
    for i=1:nc
        (a[:,i],v[:,i]) = levinson(acf[:,i], p)
    end
    return (a,v)
end


## Freely after octave's implementation, ver 3.2.4, by  jwe && jh
## skipped sparse implementation
function toeplitz{T<:Real}(c::Vector{T}, r::Vector{T}=c)
    nc = length(r)
    nr = length(c)
    res = zeros(typeof(c[1]), nr, nc)
    if nc==0 || nc==0
        return res
    end
    if r[1]!=c[1]
        ## warn
    end
    if typeof(c) <: Complex
        conj!(c)
        c[1] = conj(c[1])       # bug in julia?
    end
    ## if issparse(c) && ispsparse(r)
    data = [r[end:-1:2], c]
    for (i,start)=zip(1:nc, nc:-1:1)
        res[:,i] = data[start:start+nr-1]
    end
    return res
end

end
