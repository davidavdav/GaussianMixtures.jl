## data.jl Julia code to handle matrix-type data on disc

## report the kind of Data structure from the type instance
kind{T,S<:String}(d::Data{T,S}) = :file
kind{T}(d::Data{T,Matrix{T}}) = :matrix
Base.eltype{T}(d::Data{T}) = T

## constructor for a plain matrix.  rowvectors: data points x represented as rowvectors
Data{T}(x::Matrix{T}) = Data(Matrix{T}[x])

## constructor for a vector of files
## Data([strings], type, loadfunction)
function Data{S<:String}(files::Vector{S}, datatype::DataType, load::Function)
    Data(files, datatype, Dict(:load => load))
end

## default load function
function _load(file::String)
    load(file, "data")
end

## default size function
function _size(file::String)
    jldopen(file) do fd
        size(fd["data"])
    end
end

## courtesy compatible save for a matrix
function JLD.save(file::String, x::Matrix)
    save(file,"data", x)
end

## Data([strings], type; load=loadfunction, size=sizefunction)
function Data{S<:String}(files::Vector{S}, datatype::DataType; kwargs...) 
    all([isa((k,v), (Symbol,Function)) for (k,v) in kwargs]) || error("Wrong type of argument", args)
    d = Dict{Symbol,Function}([kwargs...])
    if !haskey(d, :load)
        d[:load] = _load
        d[:size] = _size
    end
    Data(files, datatype, d)
end

## constructor for a plain file.
Data(file::String, datatype::DataType, load::Function) = Data([file], datatype, load)
Data(file::String, datatype::DataType; kwargs...) = Data([file], datatype; kwargs...)

## is this really a shortcut?
API(d::Data, f::Symbol) = d.API[f]

function getindex(x::Data, i::Int) 
    if kind(x) == :matrix
        x.list[i]
    elseif kind(x) == :file
        x.API[:load](x.list[i])
    else
        error("Unknown kind")
    end
end

function getindex{T,VT}(x::Data{T,VT}, r::Range)
    Data{T, VT}(x.list[r], x.API)
end

## define an iterator for Data
Base.length(x::Data) = length(x.list)
Base.start(x::Data) = 0
Base.next(x::Data, state::Int) = x[state+1], state+1
Base.done(x::Data, state::Int) = state == length(x)

## This function is like pmap(), but executes each element of Data on a predestined
## worker, so that file caching at the local machine is beneficial
function dmap(f::Function, x::Data)
    if kind(x) == :file
        nx = length(x)
        w = workers()
        nw = length(w)
        worker(i) = w[1 .+ ((i-1) % nw)]
        results = cell(nx)
        getnext(i) = x.list[i]
        load = x.API[:load]
        @sync begin
            for i = 1:nx
                @async begin
                    next = getnext(i)
                    results[i] = remotecall_fetch(worker(i), s->f(load(s)), next)
                end
            end
        end
        results
    elseif kind(x) == :matrix
        pmap(f, x)
    else
        error("Unknown kind")
    end
end

## stats: compute nth order stats for array (this belongs in stats.jl)
function stats{T<:FloatingPoint}(x::Matrix{T}, order::Int=2; kind=:diag, dim=1)
    n, d = nthperm([size(x)...], dim) ## swap or not trick
    if kind == :diag
        if order == 2
            return n, vec(sum(x, dim)), vec(sumsq(x, dim))   # NumericExtensions is fast
        elseif order == 1
            return n, vec(sum(x, dim))
        else
            sx = zeros(T, order, d)
            for j=1:d
                for i=1:n
                    if dim==1
                        xi = xp = x[i,j]
                    else
                        xi = xp = x[j,i]
                    end
                    sx[1,j] += xp
                    for o=2:order
                        xp *= xi
                        sx[o,j] += xp
                    end
                end
            end
            return tuple([n, map(i->vec(sx[i,:]), 1:order)...]...)
        end
    elseif kind == :full
        order == 2 || error("Can only do covar starts for order=2")
        ## lazy implementation
        sx = vec(sum(x, dim))
        sxx = x' * x
        return n, sx, sxx
    else
        error("Unknown kind")
    end
end

## Helper functions for stats tuples:
## This relies on sum(::Tuple), which sums over the elements of the tuple. 
function +(a::Tuple, b::Tuple)
    length(a) == length(b) || error("Tuples must be of same length in addition")
    tuple(map(sum, zip(a,b))...)
end
Base.zero(t::Tuple) = map(zero, t)

## this function calls dmap as an option for parallelism
function stats(d::Data, order::Int=2; kind=:diag, dim=1)
    s = dmap(x->stats(x, order, kind=kind, dim=dim), d)
    if dim==1
        return reduce(+, s)
    else
        ## this is admittedly hairy: vertically concatenate each element of stats
        n = s[1][1]
        st = map(i->reduce(vcat, [x[i] for x in s]), 1+(1:order))
        return tuple(n, st...)
    end
end

## helper function to get summary statistics in traditional shape
function retranspose(x::Array, dim::Int)
    if dim==1
        return x'
    else
        return x
    end
end

## sum, mean, var
function Base.sum(d::Data)
    s = zero(eltype(d))
    for x in d
        s += sum(x)
    end
    return s
end

Base.sum(d::Data, dim::Int) = retranspose(stats(d,1, dim=dim)[2], dim)

function Base.mean(d::Data)
    n, sx = stats(d, 1)
    sum(sx) / (n*length(sx))
end

 function Base.mean(d::Data, dim::Int)
     n, sx = stats(d, 1, dim=dim)
     return retranspose(sx ./ n, dim)
end

function Base.var(d::Data)
    n, sx, sxx = stats(d, 2)
    n *= length(sx)
    ssx = sum(sx)                       # keep type stability...
    ssxx = sum(sxx)
    μ = ssx / n
    return (ssxx - n*μ^2) / (n - 1)
end
    
function Base.var(d::Data, dim::Int)
    n, sx, sxx = stats(d, 2, dim=dim)
    μ = sx ./ n
    return retranspose((sxx - n*μ.^2) ./ (n-1), dim)
end

function Base.cov(d::Data)
    n, sx, sxx = stats(d, 2, kind=:full)
    μ = sx ./ n
    (sxx - n*μ*μ') ./ (n-1)
end

## this is potentially very slow because it reads all file just to find out the size
function Base.size(d::Data)
    if kind(d) == :file && :size in d.API
        s = dmap(d.API[:size], d.list)
    else
        s = dmap(size, d)
    end
    nrow, ncol = s[1]
    ok = true
    for i in 2:length(s)
        ok &= s[i][2] == ncol
        nrow += s[i][1]
    end
    if !ok
        error("Inconsistent number of columns in data")
    end
    nrow, ncol
end

function Base.size(d::Data, dim::Int)
    if dim==2
        size(d[1],dim)
    else
        size(d)[dim]
    end
end

Base.collect(d::Data) = vcat([x for x in d]...)
for (f,t) in ((:float32, Float32), (:float64, Float64))
    eval(Expr(:import, :Base, f))
    @eval begin
        function ($f)(d::Data) 
            if kind(d) == :file
                api = copy(d.API)
                _load = api[:load] # local copy
                api[:load] = x -> ($f)(_load(x))
                Data(d.list, $t, api)
            elseif kind(d) == :matrix
                Data(Matrix{$t}[($f)(x) for x in d.list])
            else
                error("Unknown kind")
            end
        end
    end
end
