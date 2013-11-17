## This function computes the response of an IIR with coefficients b and a, 
## for a signal x

## it is extremely inefficient if a is given (i.e., an iir)...

## see the definition of Filter which is in filtertype.jl for practical reasons

export Filter, filter

import Base.filter

function filter{T<:Real}(x::Vector{T}, f::Filter) 
    N = length(f.c)             # length of yhist
    M = length(f.d)-1           # length of xhist
    y = vcat(f.yhist, zero(x))  # N y history
    xx = vcat(f.xhist, x)       # M x history
    if N==0 
        ## xx[1+M] is the first x valus
        ## f.d[1+M] is the first coefficient, for the currect x[i], 
        ## f.d[1] is the last coefficient, for the delayed x[i-M]
        for i=1:M+1
            y += f.d[i]*xx[i:end-M-1+i] 
        end
    else                        # slow implementation
        for n=1:length(x)
            y[n+N] = -sum(f.c .* y[n:n+N-1]) + sum(f.d .* xx[n:n+M])
        end
    end
    f.xhist=x[end-M+1:]
    f.yhist=y[end-N+1:]
    return(y[N+1:])
end
|{T<:Real}(x::Vector{T}, f::Filter) = filter(x, f)

## Array generalization, possibly in parallel
filter{T<:Real}(x::Array{T}, f::Filter) = @parallel (hcat) for i=1:size(x,2) x[:,i] | copy(f) end
|{T<:Real}(x::Array{T}, f::Filter) = filter(x, f)

##end
