## This function computes the response of an IIR with coefficients b and a, 
## for a signal x

## it is extremely inefficient if a is given (i.e., an iir)...

## see the definition of Filter which is in filtertype.jl for practical reasons

export Filter, filter

import Base.filter

function filter{T<:Real}(x::Vector{T}, f::Filter{T})
    N = length(f.c)             # length of yhist
    M = length(f.d)-1           # length of xhist
    y = vcat(f.yhist, zero(x))  # N y history
    xx = vcat(f.xhist, x)       # M x history
    s::T = 0
    for n=1:length(x)
        s=0
        for j=1:N s -= f.c[j]*y[n-1+j] end
        for j=1:M+1 s += f.d[j]*xx[n-1+j] end
        y[n+N] = s
    end
    f.xhist=x[end-M+1:]
    f.yhist=y[end-N+1:]
    return(y[N+1:])
end
|{T<:Real}(x::Vector{T}, f::Filter{T}) = filter(x, f)

## Array generalization, possibly in parallel
filter{T<:Real}(x::Array{T}, f::Filter{T}) = @parallel (hcat) for i=1:size(x,2) x[:,i] | copy(f) end
|{T<:Real}(x::Array{T}, f::Filter{T}) = filter(x, f)

##end
