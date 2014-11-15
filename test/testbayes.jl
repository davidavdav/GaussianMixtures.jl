reload("nomodule.jl")
srand(1)
xx = readdlm("test/faithful.txt")
gg = GMM(8, xx, nIter=0, kind=:full)
pp = GMMprior(gg.d, 0.1, 1.0)
vv = VGMM(gg, pp);
dd = diff(em!(copy(vv), xx))

function check()
    d = diff(em!(copy(vv), xx))
    if length(d) != length(dd)
        println("Different length L log, new d is ", d)
        return
    end
    maximum(abs(d - dd))
end

gmm = rand(GMM, 64, 26, kind=:full)
x = rand(gmm, 10000)
gmm.nx = 1000                   # fake number of data points, otherwise we get the prior
vgmm = VGMM(gmm, GMMprior(gmm.d, 0.1, 1.0))
ex = expectations(vgmm)

function prof()
    stats(vgmm, rand(10,vgmm.d), ex)
    v2 = copy(vgmm)
    Profile.clear()
    @profile em!(v2, x, nIter=1)
    Profile.print(format=:flat)
end
