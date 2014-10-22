## test most routines with randomly generated data
using NumericExtensions

for (kind, Ng) in zip((:diag, :full), (256, 16))
    println("Kind: ", kind)
    ## generate a random GMM
    gmm = rand(GMM, Ng, 26, sep=0.1, kind=kind)
    ## and generate data from the GMM
    x = rand(gmm, 100000)

    ## do some computations
    st = stats(gmm, x)
    println("nx: ", size(x,1), " sum(zeroth order stats): ", sum(st[3]))
    ll = llpg(gmm, x)
    println("avll from llpg: ", sum(log(exp(ll) * gmm.w)) / length(x))
    av = avll(gmm, x)
    println("avll direct: ", av)
    p = posterior(gmm, x)
    println("sum posterior: ", sum(p))
end
    
## and finally train a second GMM using the data
#g2 = GMM(32, x, nIter=50)
#display(means(gmm))
#display(means(g2))

