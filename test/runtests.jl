module TestingDiscreteVoronoiDiagrams
using DiscreteVoronoiDiagrams
using Test, Random

manhattan_distance(x::Real, y::Real) = abs(x - y)
manhattan_distance(x::Tuple{Real}, y::Tuple{Real}) = manhattan_distance(x[1], y[1])
@generated manhattan_distance(x::NTuple{N,Real}, y::NTuple{N,Real}) where {N} =
    Expr(:call, :+, ntuple(i -> :(manhattan_distance(x[$i], y[$i])), Val(N))...)

@testset "DiscreteVoronoiDiagrams" begin
    @testset "Utilities" begin
        let flatten = DiscreteVoronoiDiagrams.flatten,
            x = (1,2,3), r = ((-1:3), Base.OneTo(4))
            @test flatten(x) === x
            @test flatten(CartesianIndex(x)) === x
            @test flatten(CartesianIndices(r)) == r
        end
        let dist = DiscreteVoronoiDiagrams.euclidean_distance,
            dist2 = DiscreteVoronoiDiagrams.squared_euclidean_distance
            for (X,Y) in (((-1,), (3,)),
                          ((2,-3),(1,2)),
                          ((0,-2,4),(-7,1,3)))
                d2 = sum(map((x,y)-> (x - y)^2, X, Y))
                @test dist2(X, Y) == d2
                @test dist2(Y, X) == d2
                @test dist2(CartesianIndex(X), Y) == d2
                @test dist2(CartesianIndex(X), CartesianIndex(Y)) == d2
                @test dist2(X, CartesianIndex(Y)) == d2
                d = sqrt(d2)
                @test dist(X, Y) == d
                @test dist(Y, X) == d
                @test dist(CartesianIndex(X), Y) == d
                @test dist(CartesianIndex(X), CartesianIndex(Y)) == d
                @test dist(X, CartesianIndex(Y)) == d
            end
        end
        let typical_coordinates = DiscreteVoronoiDiagrams.typical_coordinates
            @test typical_coordinates(Array{Bool}(undef,0,0)) === zero(CartesianIndex{2})
            @test typical_coordinates(CartesianIndex{3}) === zero(CartesianIndex{3})
            @test typical_coordinates(Tuple{Int16,Float32}) === (zero(Int16),zero(Float32))
        end
        let nearest = DiscreteVoronoiDiagrams.nearest
            @test nearest(Int, 3) === 3
            @test nearest(Int, π) === 3
            @test nearest(Int, -sqrt(2)) === -1
            @test nearest(Int, -2//3) === -1
            @test nearest(CartesianIndex, CartesianIndex(-1,2,3)) === CartesianIndex(-1,2,3)
            @test nearest(CartesianIndex, (-π,3*sqrt(2),0x01)) === CartesianIndex(-3,4,1)
            @test nearest(CartesianIndex{3}, CartesianIndex(-1,2,3)) === CartesianIndex(-1,2,3)
            @test nearest(CartesianIndex{3}, (-π,3*sqrt(2),0x01)) === CartesianIndex(-3,4,1)
            R = CartesianIndices(((-1:3), Base.OneTo(4), (0:7)))
            @test nearest(R, CartesianIndex(-5,4,11)) === CartesianIndex(-1,4,7)
            @test nearest(R, (-π,3*sqrt(2),0x01)) === CartesianIndex(-1,4,1)
        end
        let Returns = DiscreteVoronoiDiagrams.Returns_
            @test Returns(π)() === π
            @test Returns(sqrt(2))("er"; x=33) === sqrt(2)
            @test Returns(false)(1,2) === false
        end
    end
    @testset "Building diagrams" begin
        for dims in ((42,17), (12,13,14))
            msk = zeros(Bool, dims)
            # NOTE: randomly chosen nodes must be sorted for the 2 different
            # methods to yield the same result.
            i = sort(randperm(length(msk))[1:30])
            msk[i] .= true
            # Build Voronoï diagrams with Euclidean distance.
            v1 = discrete_voronoi(msk; alg=:ref)
            v2 = discrete_voronoi(msk; alg=:best)
            @test v2 == v1
            @test discrete_voronoi!(v2, msk) == v1
            list = CartesianIndices(msk)[i]
            v3 = discrete_voronoi(dims, list; alg=:ref)
            v4 = discrete_voronoi(dims, list; alg=:best)
            @test v4 == v3
            @test discrete_voronoi!(v4, list) == v3
            @test i[v3] == v1
            # Build Voronoï diagrams with Manahattan distance.
            v1 = discrete_voronoi(msk; dist=manhattan_distance, alg=:ref)
            v2 = discrete_voronoi(msk; dist=manhattan_distance, alg=:best)
            @test v2 == v1
            list = CartesianIndices(msk)[i]
            v3 = discrete_voronoi(dims, list; dist=manhattan_distance, alg=:ref)
            v4 = discrete_voronoi(dims, list; dist=manhattan_distance, alg=:best)
            @test v4 == v3
            @test i[v3] == v1
        end
    end
end

end # module
