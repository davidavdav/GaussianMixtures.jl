
@testset "Weighted GMM" begin
    Random.seed!(42)
    d = 2
    n = 1000
    # Create random data
    x = randn(n, d)

    # Initialize a GMM
    init_gmm = GMM(3, d, kind=:diag)
    # Randomize means slightly to separate them
    init_gmm.μ = randn(3, d)
    init_gmm.Σ = ones(3, d)
    init_gmm.w = ones(3) ./ 3

    # Test 1: weights = ones should be same as no weights
    gmm_nw = copy(init_gmm)
    ll_nw = em!(gmm_nw, x, nIter=5)

    gmm_w1 = copy(init_gmm)
    w1 = ones(n)
    ll_w1 = em!(gmm_w1, x, nIter=5, weights=w1)

    @test gmm_w1.μ ≈ gmm_nw.μ atol = 1e-10
    @test gmm_w1.Σ ≈ gmm_nw.Σ atol = 1e-10
    @test gmm_w1.w ≈ gmm_nw.w atol = 1e-10
    @test ll_w1 ≈ ll_nw atol = 1e-10

    # Test 2: weights = 2 should be same as doubled data
    x2 = vcat(x, x)
    gmm_x2 = copy(init_gmm)
    ll_x2 = em!(gmm_x2, x2, nIter=5)

    gmm_w2 = copy(init_gmm)
    w2 = 2.0 * ones(n)
    ll_w2 = em!(gmm_w2, x, nIter=5, weights=w2)

    @test gmm_w2.μ ≈ gmm_x2.μ atol = 1e-10
    @test gmm_w2.Σ ≈ gmm_x2.Σ atol = 1e-10
    @test gmm_w2.w ≈ gmm_x2.w atol = 1e-10
    @test ll_w2 ≈ ll_x2 atol = 1e-10

    println("Diagonal tests passed.")

    # Test Full Covariance
    println("Testing Full Covariance...")
    init_gmm_full = GMM(3, d, kind=:full)
    init_gmm_full.μ = randn(3, d)
    # Init full covs
    for k = 1:3
        init_gmm_full.Σ[k] = GaussianMixtures.cholinv(Matrix{Float64}(I, d, d))
    end
    init_gmm_full.w = ones(3) ./ 3

    gmm_full_nw = copy(init_gmm_full)
    ll_full_nw = em!(gmm_full_nw, x, nIter=5)

    gmm_full_w1 = copy(init_gmm_full)
    ll_full_w1 = em!(gmm_full_w1, x, nIter=5, weights=w1)

    @test gmm_full_w1.μ ≈ gmm_full_nw.μ atol = 1e-10
    # For full cov, sigma is vector of upper triangulars or inverse choleskys.
    # We should compare them elementwise.
    for k = 1:3
        @test gmm_full_w1.Σ[k] ≈ gmm_full_nw.Σ[k] atol = 1e-10
    end
    @test gmm_full_w1.w ≈ gmm_full_nw.w atol = 1e-10
    @test ll_full_w1 ≈ ll_full_nw atol = 1e-10

    println("Full Covariance tests passed.")

    # Test Constructors
    println("Testing Constructors...")
    # GMM(x; weights)
    gmm_c1 = GMM(x; kind=:diag, weights=w1) # weights=1
    gmm_c_nw = GMM(x; kind=:diag)
    @test gmm_c1.μ ≈ gmm_c_nw.μ atol = 1e-10
    @test gmm_c1.Σ ≈ gmm_c_nw.Σ atol = 1e-10

    # GMM(n, x; weights) - Kmeans
    # Seed random for determinism in kmeans
    Random.seed!(42)
    gmm_k_nw = GMM(3, x; kind=:diag, nInit=1, nIter=5, method=:kmeans)

    Random.seed!(42)
    gmm_k_w1 = GMM(3, x; kind=:diag, nInit=1, nIter=5, method=:kmeans, weights=w1)

    @test gmm_k_w1.μ ≈ gmm_k_nw.μ atol = 1e-10
    @test gmm_k_w1.w ≈ gmm_k_nw.w atol = 1e-10

    # GMM(n, x; weights) - Split
    gmm_s_nw = GMM(4, x; kind=:diag, nIter=5, method=:split)
    gmm_s_w1 = GMM(4, x; kind=:diag, nIter=5, method=:split, weights=w1)
    # Checking if split method remains deterministic and identical with unit weights
    # Split method uses avll history, which should be identical.
    @test gmm_s_w1.μ ≈ gmm_s_nw.μ atol = 1e-10


    println("Constructor tests passed.")

    # Test Kmeans init disabling subsampling for weights
    # n=5, 1000 points. nneeded = 5000 > n_x.
    # Set n_x larger.
    println("Testing Large Data Kmeans Init (Subsampling Skip)...")
    nx_large = 10100
    x_large = randn(nx_large, d)
    w_large = ones(nx_large)
    # If subsampling happens with weights, it might fail or behave differently.
    # If subsampling is skipped, it should run on full data.
    # We can't easily check internal behavior without logs, but we can check it runs successfully.
    gmm_large = GMM(5, x_large; method=:kmeans, weights=w_large, nIter=1)
    @test gmm_large.n == 5
    println("Large data test passed.")

end
