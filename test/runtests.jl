module TestingDiscreteVoronoiDiagrams
using DiscreteVoronoiDiagrams
using Test, Random

@testset "DiscreteVoronoiDiagrams.jl" begin
    for dims in ((42,17), (12,13,14))
        msk = zeros(Bool, dims)
        # NOTE: randomly chosen nodes must be sorted for the 2 different
        # methods to yield the same result.
        i = sort(randperm(length(msk))[1:30])
        msk[i] .= true
        v1 = discrete_voronoi(msk; alg=:ref)
        v2 = discrete_voronoi(msk; alg=:best)
        @test v2 == v1
        list = CartesianIndices(msk)[i]
        v3 = discrete_voronoi(dims, list; alg=:ref)
        v4 = discrete_voronoi(dims, list; alg=:best)
        @test v4 == v3
        @test i[v3] == v1
    end
end

end # module
