function mfcc(x::Vector; sr=16000.0, wintime=0.025, steptime=0.01, numcep=13, lifterexp=-22, 
              sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=8000.0, 
              nbands=20, bwidth=1.0, dcttype=3, fbtype="htkmel", usecmp=false, modelorder=0)
    if (preemph!=0) 
        x = x | Filter([1, -preemph])
    end
    pspec = powspec(x, sr, wintime=wintime, steptime=steptime, dither=dither)
    aspec = audspec(pspec, sr, nbands, fbtype, minfreq, maxfreq, sumpower, bwith)
end
