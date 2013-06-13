## Freely adapted from Dan Ellis's rastamat matlab package
## We don't like matlab, which is a trademark. 

function powspec(x::Vector, sr=8000.0; wintime=0.025, steptime=0.01, dither=true)
    nwin = iround(wintime*sr)
    nstep = iround(steptime*sr)

    nfft = 1 << iceil(log2(nwin))
    window = hamming(nwin)
    noverlap = nwin - nstep
    
    y = abs2(specgram(x, nfft; sr=sr, window=window, overlap=noverlap)[1])
    y += dither * nwin

    return y
end

## Free after Octave's hamming by AW <Andreas.Weingessel@ci.tuwien.ac.at>
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

## Freely after Paul Kienzle <pkienzle@users.sf.net>

function specgram(x::Vector, n=256; sr=8000, window=hamming(n), overlap=n/2)
    if typeof(window) == Int
        window = hanning(n)
    end
    winsize = min(length(window),n)
    step = winsize - overlap

    offset = 1:step:length(x)-winsize
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
