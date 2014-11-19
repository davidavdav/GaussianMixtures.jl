## test most routines with randomly generated data
using NumericExtensions

for (gmmkind, Ng) in zip((:diag, :full), (256, 16))
    println("Kind: ", gmmkind, ", size", Ng)
    ## generate a random GMM
    gmm = rand(GMM, Ng, 26, sep=0.1, kind=gmmkind)
    ## and generate data from the GMM
    x = rand(gmm, 100000)

    ## do some computations
    st = stats(gmm, x)
    println("nx: ", size(x,1), " sum(zeroth order stats): ", sum(st[3]))
    println("avll from stats: ", st[2] / length(x))
    ll = llpg(gmm, x)
    println("avll from llpg:  ", sum(log(exp(ll) * gmm.w)) / length(x))
    av = avll(gmm, x)
    println("avll direct:     ", av)
    p, ll = posterior(gmm, x)
    println("sum posterior: ", sum(p))
end
    
## and finally train a second GMM using the data
gmm = rand(GMM, 32, 26, sep=0.1, kind=:diag)
x = rand(gmm, 100000)
g2 = GMM(32, x, nIter=50)
## what is the point of displaying?
display(means(gmm))
display(means(g2))

