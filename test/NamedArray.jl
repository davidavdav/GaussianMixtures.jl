## This is just an attempt to see if we can do named arrays

type NamedArray{T,N} <: AbstractArray{T,N}
    array::Array{T,N}
    names::NTuple{N,Vector}
    dicts::NTuple{N,Dict}
    function NamedArray(names::NTuple{N,Vector})
        array = Array(T,map(length,names))
        dicts = map(function(n) Dict(n,1:length(n)) end, names)
        println(dicts)
        new(array, names, dicts)
    end
    NamedArray(array::Array{T,N}, names::NTuple{N,Vector}, dicts::NTuple{N,Dict}) = new(array, names, dicts)
end
NamedArray(T::DataType, names::NTuple) = NamedArray{T,length(names)}(names)
function NamedArray(T::DataType, dims::Int...)
    ld = length(dims)
    names = [[string(j) for j=1:i] for i=dims]
    vec2tuple(x...) = x
    NamedArray(T, vec2tuple(names...))
end

## copy
import Base.copy
copy(A::NamedArray) = NamedArray{typeof(A[1]),length(A.names)}(copy(A.array), copy(A.names), copy(A.dicts))

import Base.print, Base.display
print(A::NamedArray) = print(A.array)
function display(A::NamedArray)
    display(typeof(A))
    display(A.names)
    display(A.array)
end

import Base.size
size(a::NamedArray) = arraysize(a.array)
size(a::NamedArray, d) = arraysize(a.array, d)

import Base.getindex, Base.to_index

getindex(A::NamedArray, s0::String) = getindex(A, A.dicts[1][s0])
getindex(A::NamedArray, s::String...) = getindex(A, map(function(t) A.dicts[t[1]][t[2]] end, zip(1:length(s), s))...)

## shameless copy from array.jl, this should prpbably be harmonized...
function getindex(A::NamedArray, I::Range1{Int}) 
    lI = length(I)
    X = similar(A.array, lI)
    if lI > 0
        copy!(X, 1, A.array, first(I), lI)
    end
    return X
end

getindex(A::NamedArray, i0::Real) = arrayref(A.array,to_index(i0))
getindex(A::NamedArray, i0::Real, i1::Real) = arrayref(A.array,to_index(i0),to_index(i1))
getindex(A::NamedArray, i0::Real, i1::Real, i2::Real) =
    arrayrefNamed(A.array,to_index(i0),to_index(i1),to_index(i2))
getindex(A::NamedArray, i0::Real, i1::Real, i2::Real, i3::Real) =
    arrayrefNamed(A.array,to_index(i0),to_index(i1),to_index(i2),to_index(i3))
getindex(A::NamedArray, i0::Real, i1::Real, i2::Real, i3::Real,  i4::Real) =
    arrayrefNamed(A.array,to_index(i0),to_index(i1),to_index(i2),to_index(i3),to_index(i4))
getindex(A::NamedArray, i0::Real, i1::Real, i2::Real, i3::Real,  i4::Real, i5::Real) =
    arrayrefNamed(A.array,to_index(i0),to_index(i1),to_index(i2),to_index(i3),to_index(i4),to_index(i5))

getindex(A::NamedArray, i0::Real, i1::Real, i2::Real, i3::Real,  i4::Real, i5::Real, I::Int...) =
    arrayref(A.array,to_index(i0),to_index(i1),to_index(i2),to_index(i3),to_index(i4),to_index(i5),I...)

import Base.setindex!

setindex!{T}(A::NamedArray{T}, x) = arrayset(A, convert(T,x), 1)

setindex!{T}(A::NamedArray{T}, x, i0::Real) = arrayset(A.array, convert(T,x), to_index(i0))
setindex!{T}(A::NamedArray{T}, x, i0::Real, i1::Real) =
    arrayset(A.array, convert(T,x), to_index(i0), to_index(i1))
setindex!{T}(A::NamedArray{T}, x, i0::Real, i1::Real, i2::Real) =
    arrayset(A.array, convert(T,x), to_index(i0), to_index(i1), to_index(i2))
setindex!{T}(A::NamedArray{T}, x, i0::Real, i1::Real, i2::Real, i3::Real) =
    arrayset(A.array, convert(T,x), to_index(i0), to_index(i1), to_index(i2), to_index(i3))
setindex!{T}(A::NamedArray{T}, x, i0::Real, i1::Real, i2::Real, i3::Real, i4::Real) =
    arrayset(A.array, convert(T,x), to_index(i0), to_index(i1), to_index(i2), to_index(i3), to_index(i4))
setindex!{T}(A::NamedArray{T}, x, i0::Real, i1::Real, i2::Real, i3::Real, i4::Real, i5::Real) =
    arrayset(A.array, convert(T,x), to_index(i0), to_index(i1), to_index(i2), to_index(i3), to_index(i4), to_index(i5))
setindex!{T}(A::NamedArray{T}, x, i0::Real, i1::Real, i2::Real, i3::Real, i4::Real, i5::Real, I::Int...) =
    arrayset(A.array, convert(T,x), to_index(i0), to_index(i1), to_index(i2), to_index(i3), to_index(i4), to_index(i5), I...)




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
