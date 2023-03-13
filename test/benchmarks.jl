module BenchmarkingDiscreteVoronoiDiagrams

using EasyFITS, LocalFilters, DiscreteVoronoiDiagrams, BenchmarkTools

if false
    f = openfits("~/data/Sphere/center_im.fits.gz");
    x = read(f[1],:,:,1,1);
    lmin, lmax = localextrema(x, 5);
    msk = (lmax .> (lmin .+ maximum(lmax - lmin)/50)) .& (lmax .== x);
    close(f)
else
    msk = readfits(Array{Bool,2}, joinpath(@__DIR__, "msk.fits.gz"))
end

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
