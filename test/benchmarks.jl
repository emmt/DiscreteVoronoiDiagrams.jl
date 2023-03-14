module BenchmarkingDiscreteVoronoiDiagrams

using DiscreteVoronoiDiagrams, BenchmarkTools

filename = joinpath(@__DIR__, "msk.txt.gz")
#decode(str::AbstractString) = map(c -> c != ' ', collect(str))
decode(str::AbstractString) = [c != ' ' for c in str]
encode(vec::AbstractVector{Bool}) = [(x ? 'x' : ' ') for x in vec]

msk = reduce(hcat, [decode(str) for str in eachline(`gzip -dc $filename`; keep=false)])

v1 = discrete_voronoi(msk; alg=:ref);
v2 = discrete_voronoi(msk; alg=:best);
@assert v2 == v1
@btime discrete_voronoi($msk; alg=:ref);
@btime discrete_voronoi($msk; alg=:best);

nodes = CartesianIndices(msk)[msk];
dims = size(msk)
v3 = discrete_voronoi(dims, nodes; alg=:ref);
v4 = discrete_voronoi(dims, nodes; alg=:best);
@assert v4 == v3
@btime discrete_voronoi($dims, $nodes; alg=:ref);
@btime discrete_voronoi($dims, $nodes; alg=:best);

end # module
