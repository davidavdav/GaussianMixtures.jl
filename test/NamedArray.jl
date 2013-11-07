## This is just an attempt to see if we can do named arrays

type NamedArray{T,N} <: AbstractArray{T,N}
    array::Array{T,N}
    names::NTuple{N,Vector}
    function NamedArray(names::NTuple{N,Vector})
        array = Array(T,map(length,names))
        new(array, names)
    end
end
NamedArray(T::DataType, names::NTuple) = NamedArray{T,length(names)}(names)
function NamedArray(T::DataType, dims::Int...)
    ld = length(dims)
    names = [[string(j) for j=1:i] for i=dims]
    vec2tuple(x...) = x
    println("Type ", T, " lenghth dims ", ld)
    NamedArray(T, vec2tuple(names...))
end

import Base.size
size(a::NamedArray) = arraysize(a.array)
size(a::NamedArray, d) = arraysize(a.array)

type Bug{d} 
    a::Int
    function Bug(dim::Int)
        d = dim
        new(a)
    end
end

type Test{T}
    t
    function Test(::Type{T})
        new(T)
    end
end
