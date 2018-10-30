using LinearAlgebra

UTriangular(a::Matrix) = UpperTriangular(a)

## NumericExtensions is no longer supported, underoptimized implementation:
function logsumexp(x::AbstractVector{T}) where {T<:AbstractFloat}
    m = maximum(x)
    log(sum(exp.(x .- m))) + m
end
logsumexp(x::Matrix{T}, dim::Integer) where {T<:AbstractFloat} = mapslices(logsumexp, x, dims=dim)

eye(n::Int) = Matrix(1.0I,n,n)
eye(::Type{Float64}, n::Int) = Matrix(1.0I,n,n)

## Also NumericExtensions' semantics of dot() is no longer supported.
function LinearAlgebra.dot(x::Matrix{T}, y::Matrix{T}) where {T<:AbstractFloat}
    size(x) == size(y) || error("Matrix sizes must match")
    dot(vec(x), vec(y))
end
function LinearAlgebra.dot(x::Matrix{T}, y::Matrix{T}, dim::Integer) where {T<:AbstractFloat}
    size(x) == size(y) || error("Matrix sizes must match")
    if dim==1
        r = zeros(T, 1, size(x,2))
        for j in 1:size(x,2)
            for i in 1:size(x,1)
                r[j] += x[i,j]*y[i,j]
            end
        end
    else
        r = zeros(T, size(x,1), 1)
        for j in 1:size(x,2)
            for i in 1:size(x,1)
                r[i] += x[i,j]*y[i,j]
            end
        end
    end
    r
end
