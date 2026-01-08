
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
end
