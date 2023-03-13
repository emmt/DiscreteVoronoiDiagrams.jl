module TestingDiscreteVoronoiDiagrams
using DiscreteVoronoiDiagrams
using Test, Random

@testset "DiscreteVoronoiDiagrams.jl" begin
    for dims in ((42,17), (12,13,14))
        msk = zeros(Bool, dims)
        i = sort(randperm(length(msk))[1:30])
        msk[i] .= true
        v1 = discrete_voronoi(msk; alg=:ref)
        list = CartesianIndices(msk)[i]
        v2 = discrete_voronoi(dims, list; alg=:ref)
        @test i[v2] == v1
    end
end

end # module
