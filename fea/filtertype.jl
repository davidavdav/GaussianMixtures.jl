## filtertype.jl.  The type used in Filter.  
## (c) 2013 David A. van Leeuwen

type Filter{T<:Real}
    a::Vector{T}
    b::Vector{T}
    c::Vector{T}
    d::Vector{T}
    xhist::Vector{T}
    yhist::Vector{T}
    function Filter(a::Vector{T}, b::Vector{T})
        xhist=zeros(length(b)-1)
        yhist=zeros(length(a)-1)
        c = reverse(a[2:])/a[1]
        d = reverse(b)/a[1]
        new(a, b, c, d, xhist, yhist)
    end
end
Filter{T<:Real}(a::Vector{T}, b::Vector{T}) = Filter{T}(a,b)
Filter{T<:Real}(a::T, b::Vector{T}) = Filter{T}([a], b)
Filter{T<:Real}(a::Vector{T}, b::T) = Filter{T}(a, [b])
Filter{T<:Real}(a::T, b::T) = Filter{T}([a], [b])
Filter{T<:Real}(b::Vector{T}) = Filter{T}([convert(T,1)], b)
Filter{T<:Real}(b::T) = Filter([b])

import Base.copy
copy{T}(f::Filter{T}) = Filter{T}(f.a, f.b)

import Base.convert
convert{T<:Real}(::Type{Filter{T}}, f::Filter) = Filter{T}(convert(Vector{T},f.a), convert(Vector{T}, f.b))
Filter{T<:Real}(::Type{T}, a, b) = convert(Filter{T}, Filter(a,b))

