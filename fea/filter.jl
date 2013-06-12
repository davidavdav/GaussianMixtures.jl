## This function computes the response of an IIR with coefficients b and a, 
## for a signal x
##module SignalProcessing

export Filter, filter

type Filter
    c::Vector
    d::Vector
    xhist::Vector
    yhist::Vector
    function Filter(a::Vector, b::Vector)
        xhist=zeros(length(b)-1)
        yhist=zeros(length(a)-1)
        c = reverse(a[2:])/a[1]
        d = reverse(b)/a[1]
        new(c, d, xhist, yhist)
    end
end
Filter(a::Number, b::Vector) = Filter([a], b)
Filter(a::Vector, b::Number) = Filter(a, [b])
Filter(b::Vector) = Filter([1.0], b)

import Base.filter

function filter(x::Vector, f::Filter) 
    N = length(f.c)             # length of yhist
    M = length(f.d)-1           # length of xhist
    y = vcat(f.yhist, similar(x))
    xx = vcat(f.xhist, x)
    for n=1:length(x)
        y[n+N] = -sum(f.c .* y[n:n+N-1]) + sum(f.d .* xx[n:n+M])
    end
    f.xhist=x[end-M+1:]
    f.yhist=y[end-N+1:]
    return(y[N+1:])
end

|(x::Vector, f::Filter) = filter(x, f)

##end
                                                              