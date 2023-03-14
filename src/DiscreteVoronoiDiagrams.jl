module DiscreteVoronoiDiagrams

export discrete_voronoi

using EasyRanges

"""
    DiscreteVoronoiDiagrams.Coordinates{N}

is the union of types representing `N`-dimensional coordinates in this package.
Currently: `N`-dimensional Cartesian indices or `N`-tuple of reals.

"""
const Coordinates{N} = Union{CartesianIndex{N},NTuple{N,Real}}

# Provide `Returns` for early versions of Julia.
if !isdefined(Base, :Returns)
    struct Returns{T}
        val::T
        Returns{T}(val) where {T} = new{T}(val)
        Returns(val::T) where {T} = new{T}(val)
    end
    (obj::Returns)(args...; kwds...) = obj.val
end

"""
    discrete_voronoi(A::AbstractArray{Bool}; alg = :best) -> inds

yields a discrete Voronoï diagram for the nodes in `A` marked as `true`. The
result is an array of (linear) indices of the nearest marked nodes for each
node of `A`. In other words:

    i = inds[j]

is the index of the marked node in `A` (such that `A[i] == true` holds) that is
the nearest to node `j` among all marked nodes in `A`. If there are no marked
nodes in `A`, the result is an array filled with `firstindex(A) - 1`.

Keyword `dist` may be used to specify another distance than the (squared)
Euclidean distance to build the diagram. The distance function must be
type-stable which yields a non-negative distance when called as `dist(a,b)`
with `a` and `b` two `N`-tuple of real-valued coordinates. In case of tie, the
first in the lexicographic order of `A` is favored.

"""
function discrete_voronoi(A::AbstractArray{Bool};
                          alg::Symbol = :best, kwds...)
    discrete_voronoi(Val(alg), A; kwds...)
end

function discrete_voronoi(alg::Val,
                          A::AbstractArray{Bool};
                          kwds...)
    discrete_voronoi!(alg, similar(A, Int), A; kwds...)
end

"""
    discrete_voronoi(dims, A) -> inds

yields a `N`-dimensional discrete Voronoï diagram of size `dims` for the nodes
whose `N`-dimensional coordinates are the elements of `A`. The result is an
array of (linear) indices such that:

    i = inds[j]

is the index of the node in `A` that is nearest to node `j` among all nodes in
`A`. The coordinates of the `i`-th node in `A` is simply `A[i]` while the
coordinates of the `j`-th node in the Voronoï diagram are its Cartesian
coordinates the array `inds` representing the diagram. If `A` is empty, the
result is an array filled with `firstindex(A) - 1`.

Keyword `dist` may be used to specify another distance than the (squared)
Euclidean distance to build the diagram. In case of tie, the first in the
lexicographic order of `A` is favored.

"""
function discrete_voronoi(dims::NTuple{N,Integer},
                          A::AbstractVector{<:Coordinates{N}};
                          alg::Symbol = :best, kwds...) where {N}
    discrete_voronoi(Val(alg), dims, A; kwds...)
end

function discrete_voronoi(alg::Val, dims::NTuple{N,Integer},
                          A::AbstractVector{<:Coordinates{N}}; kwds...) where {N}
    discrete_voronoi!(alg, Array{Int,N}(undef, dims), A; kwds...)
end

"""
    discrete_voronoi!(inds, A) -> inds

overwrites `inds` with the `N`-dimensional discrete Voronoï diagram for the
nodes specified by `A` which may be:

- a boolean array with the same indices as `inds` where the centers of the
  Voronoï domains are indicated by the true values in `A`;

- a vector of the `N`-dimensional coordinates of the centers of the
  Voronoï domains.

This in-place version of [`discrete_voronoi`](@ref) may be useful to deal with
a diagram `inds` with non-standard indices (e.g. arrays from the `OffsetArrays`
package).

See [`discrete_voronoi`](@ref) for more details and for allowed keywords.

"""
function discrete_voronoi!(inds::AbstractArray{Int,N},
                           A::Union{AbstractArray{Bool,N},
                                    AbstractVector{<:Coordinates{N}}};
                           alg::Symbol = :best, kwds...) where {N}
    discrete_voronoi!(Val(alg), inds, A; kwds...)
end

function discrete_voronoi!(alg::Val,
                           inds::AbstractArray{Int,N},
                           A::Union{AbstractArray{Bool,N},
                                    AbstractVector{<:Coordinates{N}}};
                           dist::Function = squared_euclidean_distance) where {N}
    # Check argument so that `unsafe_build!` can be called safely.
    !(A isa AbstractArray{Bool,N}) || axes(inds) == axes(A) ||
        throw(DimensionMismatch("arrays have different indices"))

    # Determine the type to use for the distance between nodes.
    T = infer_distance_type(dist, A)

    # Call the builder.
    return unsafe_discrete_voronoi!(alg, dist, T, inds, A)
end

# Reference method. Slow: takes ~ 1.263510 seconds on a 290×290 image.
function unsafe_discrete_voronoi!(::Val{:ref}, dist::Function, ::Type{T},
                                  inds::AbstractArray{Int,N},
                                  A::Union{AbstractArray{Bool,N},
                                           AbstractVector{<:Coordinates{N}}}) where {T,N}
    # Initialize the result with an invalid index to the nearest neighbor in
    # the source.
    fill!(inds, invalid_node_index(A))
    length(A) ≥ 1 || return inds # quick return, nothing else to do

    # Instantiate `dmin` workspace to store min. distance to nearest neighbor.
    dmin = fill!(similar(inds, T), typemax(T))

    # Build the Voronoï diagrame naively (but surely).
    R = CartesianIndices(inds)
    @inbounds for i in eachindex(A)
        if A isa AbstractArray{Bool,N}
            A[i] || continue
            I = R[i] # Cartesian index of i
        else
            I = A[i] # coordinates of i-th node
        end
        for j in eachindex(inds)
            J = R[j] # Cartesian index of j
            d = dist(flatten(I), flatten(J))::T
            if d < dmin[j] || (d == dmin[j] && i < inds[j])
                dmin[j] = d
                inds[j] = i
            end
        end
    end
    return inds
end

# Best method (so far). Fast: takes ~ 0.014269 seconds on a 290×290 image and
# yields same result as reference method
function unsafe_discrete_voronoi!(::Val{:best}, dist, ::Type{T},
                                  inds::AbstractArray{Int,N},
                                  A::Union{AbstractArray{Bool,N},
                                           AbstractVector{<:Coordinates{N}}}) where {T,N}
    # Initialize the result with an invalid index to the nearest neighbor in
    # the source.
    unset = invalid_node_index(A) # to mark unset indices
    fill!(inds, unset)
    length(A) ≥ 1 || return inds # quick return, nothing else to do

    # Instantiate `dmin` workspace to store min. distance to nearest neighbor.
    dmin = fill!(similar(inds, T), typemax(T))

    # Instantiate `age` workspace to avoid examining a node more than once.
    stage = 0 # initial stage
    age = fill!(similar(inds, typeof(stage)), stage)

    # Create objects for fast linear <-> Cartesian coordinates conversion.
    L = LinearIndices(inds)    # Cartesian -> linear index map
    R = CartesianIndices(inds) # linear -> Cartesian index map
    B = CartesianIndices(ntuple(i -> -1:1, Val(N))) # neighborhood

    # Create queues and initialize secondary queue.
    n = length(inds)
    Q1 = sizehint!(Tuple{Int,Int}[], n) # primary queue for current nodes to examinate
    Q2 = sizehint!(Tuple{Int,Int}[], n) # secondary queue for next nodes to examinate
    @inbounds for i in eachindex(A)
        if A isa AbstractArray{Bool,N}
            A[i] || continue
            j = i # at this stage, the nearest node is the node itself
        else
            I = A[i] # coordinates of i-th node
            J = nearest(R, I) # position in R that is the nearest to I
            j = L[J] # linear index of j
        end
        push!(Q2, (i,j))
    end

    # Iteratively process queue of pair of nodes to examine.
    @inbounds while !isempty(Q2)
        # Swap the primary and secondary queues and empty the secondary one.
        Q1, Q2 = Q2, Q1
        empty!(Q2)
        # Examine each node in the primary queue and enqueue their neighbors if
        # they are part of the Voronoï domain of the current central node.
        prev = unset
        for (i,j) in Q1
            if i != prev
                # Central node has changed.
                prev = i
                stage += 1
            end
            I = if A isa AbstractArray{Bool,N}
                R[i] # Cartesian index of i
            else
                A[i] # coordinates of i-th node
            end
            J = R[j] # Cartesian index of j
            d = dist(flatten(I), flatten(J))::T
            if d < dmin[j] || (d == dmin[j] && i < inds[j])
                # Node i is the nearest neighbor of node j so far. Update
                # diagram and push nodes of the neighborhood of j into the
                # queue for later examination.
                inds[j] = i
                dmin[j] = d
                @inbounds for K ∈ @range R ∩ (B + J) # for each node in the region
                    k = L[K] # linear index of K
                    if inds[k] != i && age[k] < stage
                        # Node k is not already part of the Voronoï domain of
                        # node i and has not yet been pushed in the queue at
                        # this stage.
                        age[k] = stage
                        push!(Q2, (i,k))
                    end
                end
            end
        end
    end
    return inds
end

invalid_node_index(A) = firstindex(A) - 1

"""
    DiscreteVoronoiDiagrams.flatten(x) -> tup

yields a `N`-tuple of coordinates given an instance `x` of
[`DiscreteVoronoiDiagrams.Coordinates{N}`](@ref) or a `N`-tuple of ranges given
an instance `x` or `CartesianIndices{N}`.

"""
flatten(x::Tuple{Vararg{Real}}) = x
flatten(x::CartesianIndex) = Tuple(x)
flatten(x::CartesianIndices) = x.indices

"""
    DiscreteVoronoiDiagrams.euclidean_distance(a, b) -> d

yields the Euclidean distance between two nodes `a` and `b` specified as
`N`-dimensional coordinates, that is instances of
[`DiscreteVoronoiDiagrams.Coordinates{N}`](@ref).

"""
euclidean_distance(a, b) = sqrt(squared_euclidean_distance(a, b))

"""
    DiscreteVoronoiDiagrams.squared_euclidean_distance(a, b) -> d

yields the squared Euclidean distance between two nodes `a` and `b` specified
as `N`-dimensional coordinates, that is instances of
`DiscreteVoronoiDiagrams.Coordinates{N}`.

"""
squared_euclidean_distance(a::Real, b::Real) = (b - a)^2
squared_euclidean_distance(a::Tuple{Real}, b::Tuple{Real}) =
    squared_euclidean_distance(a[1], b[1])
squared_euclidean_distance(a::Coordinates{N}, b::Coordinates{N}) where {N} =
    squared_euclidean_distance(flatten(a), flatten(b))
@generated squared_euclidean_distance(a::NTuple{N,Real}, b::NTuple{N,Real}) where {N} =
    Expr(:call, :+, ntuple(i -> :(squared_euclidean_distance(a[$i], b[$i])), Val(N))...)

function infer_distance_type(dist::Function,
                             A::Union{AbstractArray{Bool,N},
                                      AbstractVector{<:Coordinates{N}}}) where {N}
    x = typical_coordinates(A)
    y = typical_coordinates(CartesianIndex{N})
    return typeof(dist(flatten(x), flatten(y)))
end

typical_coordinates(A::AbstractArray{Bool,N}) where {N} =
    typical_coordinates(CartesianIndex{N})
typical_coordinates(A::AbstractVector{<:Coordinates{N}}) where {N} =
    typical_coordinates(eltype(A))

# NOTE: `CartesianIndex{N}(x)` works for converting `x` to a Cartesian index,
#       while `convert(T,x)` works for converting `x` to a tuple of type `T`.
typical_coordinates(::Type{<:CartesianIndex{N}}) where {N} =
    CartesianIndex{N}(ntuple(Returns(0), Val(N)))
typical_coordinates(::Type{T}) where {N,T<:NTuple{N,Real}} =
    convert(T, ntuple(Returns(0), Val(N)))::T

"""
    DiscreteVoronoiDiagrams.nearest(T::Type, x) -> y

yields the value `y` of type`T` that is the nearest to `x`. `T` can be an
integer type and `x` a real value, or `T` can be `CartesianIndex{N}` and `x` a
`N`-dimensional Cartesian index or a `N`-tuple of reals.

"""
nearest(::Type{T}, x::T) where {T<:Integer} = x
nearest(::Type{T}, x::Integer) where {T<:Integer} = convert(T, x)::T
nearest(::Type{T}, x::Real) where {T<:Integer} = round(T, x)::T
nearest(::Type{T}, x::Irrational) where {T<:Integer} = round(T, float(x))::T

nearest(::Type{CartesianIndex{N}}, x::CartesianIndex{N}) where {N} = x
nearest(::Type{CartesianIndex{N}}, x::NTuple{N,Real}) where {N} =
    CartesianIndex(map(nearest_int, x))

"""
    DiscreteVoronoiDiagrams.nearest(R::CartesianIndices, I) -> J

yields the Cartesian index `J` inside the `N`-dimensional region `R` that is
the nearest to `I`. `I` may be a `N`-dimensional Cartesian index or a `N`-tuple
of reals.

"""
nearest(R::CartesianIndices{N}, I::Coordinates{N}) where {N} =
    CartesianIndex(map(nearest_int, flatten(R), flatten(I)))

nearest_int(x::Real) = nearest(Int, x)
nearest_int(r::AbstractUnitRange{<:Integer}, x::Real) =
    clamp(nearest_int(x), Int(first(r))::Int, Int(last(r))::Int)

end
