## data.jl Test some functionality of the Data type
## (c) 2015 David A. van Leeuwen

@testset "data.jl" begin

	for i = 1:10
	    save("$i.jld", "data", randn(10000,3))
	end
	x = Matrix{Float64}[load("$i.jld", "data") for i=1:10]

	g = rand(GMM, 2, 3)
	d = Data(x)
	dd = Data(["$i.jld" for i=1:10], Float64)

	f1(gmm, data) = GaussianMixtures.dmapreduce(x->stats(gmm, x), +, data)
	f2(gmm, data) = reduce(+, map(x->stats(gmm, x), data))

	sleep(1)
	println(f2(g,dd))

	s = stats(g, collect(d))

	@test isapprox(s, f1(g,d))
	@test isapprox(s, f1(g,dd))
	@test isapprox(s, f2(g,d))
	@test isapprox(s, f2(g,dd))


	@test isapprox(s, stats(g,d))
	@test isapprox(s, stats(g,dd))

	# parallel
	addprocs(1)
	@everywhere using GaussianMixtures

	@test isapprox(s, stats(g,d, parallel=true))
	@test isapprox(s, stats(g,dd, parallel=true))

	rmprocs(workers())
    
end

