
@testset "train.jl" begin

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
        println("avll from llpg:  ", sum(log.(exp.(ll) * gmm.w)) / length(x))
        av = avll(gmm, x)
        println("avll direct:     ", av)
        p, ll = gmmposterior(gmm, x)
        println("sum posterior: ", sum(p))
    end

    ## and finally train a second GMM using the data
    for gmmkind in [:diag, :full]
        ## what is the point of displaying?
        gmm = rand(GMM, 32, 26, sep=0.1, kind=gmmkind)
        ## display(means(gmm))
        x = rand(gmm, 100000)
        for method in [:split, :kmeans]
            println("kind $gmmkind, method $method")
            g2 = GMM(32, x, nIter=50, kind=gmmkind, method=method)
            ## display(means(g2))
            ## do another iteration of em!
            em!(g2, x)
        end
    end

    ## Check that the uninitialized constructor doesn't trigger an error.
    for gmmkind in [:diag, :full]
        GMM(32, 26; kind=gmmkind)
    end

    @test_broken false



end

## test most routines with randomly generated data

