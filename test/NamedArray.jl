## This is just an attempt to see if we can do named arrays

type NamedArray{T,N} <: AbstractArray{T,N}
    array::Array{T,N}
    names::Vector{Vector}
    function NamedArray(names::Vector)
        @assert N==length(names) > 0
        @assert isa(names[1], Vector)
        array = zeros(T,map(length,names)...)
        new(array, names)
    end
end
#NamedArray(T::DataType, names::Vector) = NamedArray{T,length(names)}(names)
function NamedArray(T::DataType, dims::Int...)
    ld = length(dims)
    name = [[string(j) for j=1:i] for i=dims]
    a = Array(T,dims...)
    println("Type ", T, " lenghth dims ", ld)
    NamedArray(a, name)
#end

type Bug{d} 
    a::Int
    function Bug(dim::Int)
        d = dim
        new(a)
    end
end

type Test
    t::DataType
    function Test(t::DataType)
        new(t)
    end
end
