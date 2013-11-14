## Miscelaneous functions that I miss in julia

module Misc

export nrow, ncol, sortperm, znorm, znorm!

nrow(x::Array) = size(x,1)
ncol(x::Array) = size(x,2)

znorm(x::Array, dim::Int=1) = broadcast(/, broadcast(-, x, mean(x, dim)), std(x, dim))
znorm!(x::Array, dim::Int=1) = broadcast!(/, x, broadcast!(-, x, x, mean(x, dim)), std(x, dim))

import Base.Sort.sortperm

sortperm(a::Array,dim::Int) = mapslices(sortperm, a, dim)

end
