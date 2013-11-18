## rasta.jl
## (c) 2013 David van Leeuwen
## recoded from Dan Ellis's rastamat package

module Rasta

## Freely adapted from Dan Ellis's rastamat matlab package.  We've kept routine names the same, but the interface has changed a bit. 
## We don't like matlab, which is a trademark. 

## we haven't implemented rasta filtering, yet, in fact.  These routines are a minimum for 
## encoding HTK-style mfccs

export powspec, audspec, fft2barkmx, fft2melmx, hz2bark, hz2mel, mel2hz, postaud, dolpc, lpc2cep, spec2cep, lifter

using SignalProcessing

# powspec tested against octave with simple vectors
function powspec{T<:FloatingPoint}(x::Vector{T}, sr::FloatingPoint=8000.0; wintime=0.025, steptime=0.01, dither=true)
    nwin = iround(wintime*sr)
    nstep = iround(steptime*sr)

    nfft = 1 << iceil(log2(nwin))
    window = hamming(nwin)      # overrule default in specgram which is hanning in Octave
    noverlap = nwin - nstep
    
    y = abs2(specgram(x * (1<<15), nfft; sr=sr, window=window, overlap=noverlap)[1])
    y += dither * nwin

    return y
end

# audspec tested against octave with simple vectors for all fbtypes
function audspec{T<:FloatingPoint}(x::Array{T}, sr::FloatingPoint=16000.0; nfilts=iceil(hz2bark(sr/2)), fbtype="bark", 
                 minfreq=0, maxfreq=sr/2, sumpower=true, bwidth=1.0)
    (nfreqs,nframes)=size(x)
    nfft = 2(nfreqs-1)
    if fbtype=="bark"
        wts = fft2barkmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
    elseif fbtype=="mel"
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
    elseif fbtype=="htkmel"
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq, htkmel=true, constamp=true)
    elseif fbtype=="fcmel"
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq, htkmel=true, constamp=false)
    else
        ## error
    end
    wts = wts[:,1:nfreqs]
    if sumpower
        return wts * x
    else
        return (wts * sqrt(x)).^2
    end
end

function fft2barkmx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=0, maxfreq=sr/2)
    hnfft = nfft>>1
    minbark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - minbark
    wts=zeros(nfilts, nfft)
    stepbark = nyqbark/(nfilts-1)
    binbarks=hz2bark([0:hnfft]*sr/nfft)
    for i=1:nfilts
        midbark = minbark + (i-1)*stepbark
        lof = (binbarks - midbark)/width - 0.5
        hif = (binbarks - midbark)/width + 0.5
        logwts = min(0, min((hif, -2.5lof)))
        wts[i,1:1+hnfft] = 10.0.^logwts
    end
    return wts
end

function hz2bark(f::Real)
    return 6asinh(f/600)
end

function fft2melmx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=0.0, maxfreq=sr/2, htkmel=false, constamp=false)
    wts=zeros(nfilts, nfft)
    # Center freqs of each DFT bin
    fftfreqs = [0:nfft-1]/nfft*sr; 
    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfreq, htkmel); 
    maxmel = hz2mel(maxfreq, htkmel);
    binfreqs = mel2hz(minmel+[0:(nfilts+1)]/(nfilts+1)*(maxmel-minmel), htkmel);
##    binbin = iround(binfrqs/sr*(nfft-1));
    
    for i=1:nfilts
        fs = binfreqs[i+(0:2)]
        # scale by width
        fs = fs[2] + width*(fs-fs[2])
        # lower and upper slopes for all bins
        loslope = (fftfreqs - fs[1])/diff(fs[1:2])
        hislope = (fs[3] - fftfreqs)/diff(fs[2:3])
        # then intersect them with each other and zero
        wts[i,:] = max(0, min(loslope,hislope))
    end
    
    if !constamp
        ## unclear what this does... 
        ## Slaney-style mel is scaled to be approx constant E per channel
        wts = broadcast(*, 2/((binfreqs[3:nfilts+2]) - binfreqs[1:nfilts]), wts)
    end
    # Make sure 2nd half of DFT is zero
    wts[:,(nfft>>1)+1:nfft] = 0
    return wts
end

function hz2mel{T<:Real}(f::Vector{T}, htk=false)
    if htk
        return 2595 * log10(1+f/700)
    else
        f0 = 0.0
        fsp = 200/3
        brkfrq = 1000.0
        brkpt = (brkfrq - f0) / fsp
        logstep = exp(log(6.4)/27)
        linpts = f .< brkfrq
        z = zeros(size(f))      # prevent InexactError() by making these Float64
        z[find(linpts)] = f[find(linpts)]/brkfrq ./ log(logstep)
        z[find(!linpts)] = brkpt + log(f[find(!linpts)]/brkfrq) ./ log(logstep)
    end
    return z
end
hz2mel(f::Number, htk=false)  = hz2mel([f], htk)[1]

function mel2hz{T<:Real}(z::Array{T}, htk=false) 
    if htk
        f = 700*(10.^(z/2595)-1)
    else
        f0 = 0.0
        fsp = 200/3
        brkfrq = 1000.0
        brkpt = (brkfrq - f0) / fsp
        logstep = exp(log(6.4)/27)
        linpts = z .< brkpt
        f = similar(z)
        f[linpts] = f0 + fsp*z[linpts]
        f[!linpts] = brkfrq*exp(log(logstep)*(z[!linpts] - brkpt))
    end
    return f
end

function postaud{T<:FloatingPoint}(x::Array{T}, fmax::Real, fbtype="bark", broaden=false)
    (nbands,nframes) = size(x)
    nfpts = nbands+2broaden
    if fbtype=="bark"
        bandcfhz = bark2hz(linspace(0, hz2bark(fmax), nfpts))
    elseif fbtype=="mel"
        bandcfhz = mel2hz(linspace(0, hz2mel(fmax), nfpts))
    elseif fbtype=="htkmel" || fbtype=="fcmel"
        bandcfhz = mel2hz(linspace(0, hz2mel(fmax,1), nfpts),1);
    else
        ## error
    end
    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[1+broaden:nfpts-broaden];
    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz.^2
    ftmp = fsq + 1.6e5
    eql = ((fsq./ftmp).^2) .* ((fsq + 1.44e6)./(fsq + 9.61e6))
    # weight the critical bands
    z = broadcast(*, eql, x)
    # cube root compress
    z .^= 0.33
    # replicate first and last band (because they are unreliable as calculated)
    if broaden
        z = z[[1,1:nbands,nbands],:];
    else
        z = z[[2,2:(nbands-1),nbands-1],:]
    end
    return z
end
    
function dolpc{T<:FloatingPoint}(x::Array{T}, modelorder::Int=8) 
    (nbands, nframes) = size(x)
    r = real(ifft(vcat(x, x[[nbands-1:-1:2],:]), 1)[1:nbands,:])
    # Find LPC coeffs by durbin
    (y,e) = levinson(r, modelorder)
    # Normalize each poly by gain
    y = broadcast(/, y, e)'
end

function lpc2cep{T<:FloatingPoint}(a::Array{T}, ncep::Int=0) 
    (nlpc, nc) = size(a)
    order = nlpc-1
    if ncep==0
        ncep=nlpc
    end
    c = zeros(ncep, nc)
    # Code copied from HSigP.c: LPC2Cepstrum
    # First cep is log(Error) from Durbin
    c[1,:] = -log(a[1,:])
    # Renormalize lpc A coeffs
    a = broadcast(/, a, a[1,:])
    for n=2:ncep
        sum=0.0
        for m=2:n
            sum += (n-m)*a[m,:] .* c[n-m+1,:]
        end
        c[n,:] = -a[n,:] - sum/(n-1)
    end
    return c
end

function spec2cep{T<:FloatingPoint}(spec::Array{T}, ncep::Int=13, dcttype::Int=2)
    (nr, nc) = size(spec)
    dctm = zeros(typeof(spec[1]), ncep, nr)
    if 1 < dcttype < 4          # type 2,3
        for i=1:ncep
            dctm[i,:] = cos((i-1)*[1:2:2nr-1]π/(2nr)) * sqrt(2/nr)
        end
        if dcttype==2
            dctm[1,:] /= sqrt(2)
        end
    elseif dcttype==4           # type 4
        for i=1:ncep
            dctm[i,:] = 2cos((i-1)*[1:nr]π/(nr+1))
            dctm[i,1] += 1
            dctm[i,nr] += (-1)^(i-1)
        end
        dctm /= 2(nr+1)
    else                        # type 1
        for i=1:ncep
            dctm[i,:] = cos((i-1)*[0:nr-1]π/(nr-1)) / (nr-1)
        end
        dctm[:,[1,nr]] /= 2
    end
    return dctm*log(spec)
end

function lifter{T<:FloatingPoint}(x::Array{T}, lift::Real=0.6, invs=false)
    (ncep, nf) = size(x)
    if lift==0
        return x
    end
    if lift>0
        if lift>10
            ## error
        end
        liftw = [1, [1:ncep-1].^lift]
    else
        # Hack to support HTK liftering
        if !isa(lift, Int)
            ## error
        end
        liftw = [1, (1 + lift/2*sin([1:ncep-1]π/lift))]
    end
    if invs
        liftw = 1./liftw
    end
    return broadcast(*, x, liftw)
end


end
