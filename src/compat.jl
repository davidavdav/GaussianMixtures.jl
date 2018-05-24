## bugs in v0.3 and compatibility
import Base.LinAlg.AbstractTriangular
UTriangular(a::Matrix) = UpperTriangular(a)

## NumericExtensions is no longer supported, underoptimized implementation:
function logsumexp{T<:AbstractFloat}(x::AbstractVector{T})
    m = maximum(x)
    log.(sum(exp.(x .- m))) + m
end
logsumexp{T<:AbstractFloat}(x::Matrix{T}, dim::Integer) = mapslices(logsumexp, x, dim)

## Also NumericExtensions' semantics of dot() is no longer supported.
function Base.dot{T<:AbstractFloat}(x::Matrix{T}, y::Matrix{T})
    size(x) == size(y) || error("Matrix sizes must match")
    dot(vec(x), vec(y))
end
function Base.dot{T<:AbstractFloat}(x::Matrix{T}, y::Matrix{T}, dim::Integer)
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

## this we need for xμTΛxμ!
#Base.A_mul_Bc!(A::StridedMatrix{Float64}, B::AbstractTriangular{Float32}) = A_mul_Bc!(A, convert(AbstractMatrix{Float64}, B))
#Base.A_mul_Bc!(A::Matrix{Float32}, B::AbstractTriangular{Float64}) = A_mul_Bc!(A, convert(AbstractMatrix{Float32}, B))
## this for diagstats
#Base.BLAS.gemm!(a::Char, b::Char, alpha::Float64, A::Matrix{Float32}, B::Matrix{Float64}, beta::Float64, C::Matrix{Float64}) = Base.BLAS.gemm!(a, b, alpha, float64(A), B, beta, C)
