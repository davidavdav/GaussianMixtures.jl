## test VB GMM for the standard example data
x = readdlm("faithful.txt")
## only do k-means
g = GMM(8, x, kind=:full, nIter=0)
p = GMMprior(g.d, 1.0, 1.0)
v = VGMM(g, p)
em!(v, x)
## show the reduction in Gaussians
println(history(v))
## we have no pretty printing yet...
for f in [:α, :β, :m, :ν, :W] 
    println(f, " = ", getfield(v, f))
end
    
