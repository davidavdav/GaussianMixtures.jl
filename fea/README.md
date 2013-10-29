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
 - `levinson(acf::Vector, p::Int)`
 - `toeplitz(c::Vector, r::Vector=c)`
