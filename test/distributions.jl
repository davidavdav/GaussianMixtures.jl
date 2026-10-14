@testset "Distributions" begin
    mm = MixtureModel([
            MvNormal(Float64[0, 0], Diagonal(Float64[1, 2])),
            MvNormal(Float64[3, 3], Diagonal(Float64[9, 7])),
        ],
        [0.2, 0.8]
    )

    gmm = GMM(mm)

    @test probs(mm) == weights(gmm)
    @test mean(component(mm, 1)) == means(gmm)[1, :]
    @test mean(component(mm, 2)) == means(gmm)[2, :]
    @test cov(component(mm, 1)).diag ≈ covars(gmm)[1, :]
    @test cov(component(mm, 2)).diag ≈ covars(gmm)[2, :]

    mm2 = MixtureModel(gmm)

    @test probs(mm2) == weights(gmm)
    @test mean(component(mm2, 1)) == means(gmm)[1, :]
    @test mean(component(mm2, 2)) == means(gmm)[2, :]
    @test cov(component(mm2, 1)).diag ≈ covars(gmm)[1, :]
    @test cov(component(mm2, 2)).diag ≈ covars(gmm)[2, :]

    mm = MixtureModel([
            MvNormal([0, 0], [4 2 ; 2 4]),
            MvNormal([3, 3], [9 7 ; 7 9]),
        ],
        [0.2, 0.8]
    )

    gmm = GMM(mm)

    @test probs(mm) == weights(gmm)
    @test mean(component(mm, 1)) == means(gmm)[1, :]
    @test mean(component(mm, 2)) == means(gmm)[2, :]
    @test cov(component(mm, 1)) ≈ covars(gmm)[1]
    @test cov(component(mm, 2)) ≈ covars(gmm)[2]

    mm2 = MixtureModel(gmm)

    @test probs(mm2) == weights(gmm)
    @test mean(component(mm2, 1)) == means(gmm)[1, :]
    @test mean(component(mm2, 2)) == means(gmm)[2, :]
    @test cov(component(mm2, 1)) ≈ covars(gmm)[1]
    @test cov(component(mm2, 2)) ≈ covars(gmm)[2]
end