

@testset "bayes.jl" begin

	x = dataset("datasets","faithful")  # RDatasets
	y = convert(Vector{Float64},x.Waiting)

	# test standard run 
	g = GMM(2,y)
	grps = sortperm(g.w)
	@test all( isapprox.(g.w[grps], [0.36; 0.639],atol=0.01) )
	@test all( isapprox.(g.μ[grps], [54.61; 80.09],atol=0.01) )
	@test all( isapprox.(sqrt.(g.Σ[grps]), [5.867; 5.871],atol=0.01) )

	## only do k-means
	x = Matrix(x)
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
end
