## filtertype.jl.  The type used in Filter.  
## (c) 2013 David A. van Leeuwen

type Filter
    a::Vector
    b::Vector
    c::Vector
    d::Vector
    xhist::Vector
    yhist::Vector
    function Filter(a::Vector, b::Vector)
        xhist=zeros(length(b)-1)
        yhist=zeros(length(a)-1)
        c = reverse(a[2:])/a[1]
        d = reverse(b)/a[1]
        new(a, b, c, d, xhist, yhist)
    end
end
Filter(a::Number, b::Vector) = Filter([a], b)
Filter(a::Vector, b::Number) = Filter(a, [b])
Filter(b::Vector) = Filter([1.0], b)

import Base.copy
copy(f::Filter) = Filter(f.a, f.b)

