## Miscelaneous functions that I miss in julia

module Misc

export nrow, ncol, sortperm

nrow(x::Array) = size(x,1)
ncol(x::Array) = size(x,2)

import Base.Sort.sortperm

sortperm(a::Array,dim::Int) = mapslices(sortperm, a, dim)

end
