function mfcc(x::Vector; sr=16000.0, numcep=13, lifterexp=-22, sumpower=false, preemph=0.97,
              dither=false, minfreq=0.0, maxfreq=8000.0, nbands=20, bwidth=1.0, dcttype=3,
              fbtype="htkmel", usecmp=false, modelorder=0)
    if (preemph!=0) 
        filter!(x, [1 -preemph], 1)
    end
end
