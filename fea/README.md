SignalProcessing
================

This module contains some signalprocessing routines necessary for feature extraction for the most common features (MFCC, PLP) in speech processing. 

Most of this is re-coded from octave 3.2.4.  

Windowing (tapers)
 - `hamming(n::Int)`
 - `hamming(n::Int)`

Spectogram
 - `specgram(x::Vector, n::Int=256; sr=8000, window=hamming(n), overlap=n/2)`
    - computes spectogram for data in `x` over window of size `n` overlapping `overlap` points

Miscelaneous

 - `levinson(acf::Vector, p::Int)`
 - `toeplitz(c::Vector, r::Vector=c)`


Rasta
=====

This module contains the main routines for MFCC and PLP extraction.  Even though it is called Rasta, rasta processing hasn't been implemented yet. 

Most of this is re-coded from Dan Ellis's rastamat package.  We like Dan Ellis's work.  

Spectra
 - `powspec(x::Vector, sr=8000.0; wintime=0.025, steptime=0.01, dither=true)`
 - `audspec(x::Array, sr=16000.0; nfilts=iceil(hz2bark(sr/2)), fbtype="bark", 
                 minfreq=0, maxfreq=sr/2, sumpower=true, bwidth=1.0)`

Filterbank
 - `fft2barkmx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=0, maxfreq=sr/2)`
 - `hz2bark(f)`
 - `fft2melmx(nfft::Int, nfilts::Int; sr=8000.0, width=1.0, minfreq=0.0, maxfreq=sr/2, htkmel=false, constamp=false)`
 - `hz2mel(f::Vector, htk=false)`
 - `mel2hz(z, htk=false)`

PLP
 - `postaud(x::Array, fmax::Number, fbtype="bark", broaden=false)`
 - `dolpc(x::Array, modelorder=8)`
 - `lpc2cep(a::Array, ncep=0)`
 - `spec2cep(spec::Array, ncep::Int=13, dcttype::Int=2)`

Postprocessing
 - `lifter(x::Array, lift::Number=0.6, invs=false)`
 - `deltas(x, w::Int=9)`

Features
========

This module computes the actual features, using the above modules.  

Again, most of this is re-coded from Dan Ellis's rastamat package.  

Note that `mfcc()` has many parameters, but most of these are set to defaults that _should_ mimick HTK default parameter (untested). 

Feature extraction
 - `mfcc(x::Vector, sr::Number=16000.0; wintime=0.025, steptime=0.01, numce
p=13, lifterexp=-22, 
              sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=s
r/2, 
              nbands=20, bwidth=1.0, dcttype=3, fbtype="htkmel", usecmp=false, m
odelorder=0)`

Feature warping, or short-time Gaussianization (Jason Pelecanos)
 - `warp(x::Array, w=399)`

Shifted-Delta-Cepstra (features for spoken language recogntion)
 - `sdc(x::Array, n::Int=7, d::Int=1, p::Int=3, k::Int=7)`


Misc
====

I felt I missed some useful routines from R.  With matrices, I have to think in terms of `rows' and `columns'.  I need to think of `data runs down, reatures run right' and those kind of things. 

 - `nrow()`
 - `ncol()`

furtherm extended `Base.Sort.sortperm()` to accept `Array`. 
 - `sortperm(a::Array,dim::Int)`
