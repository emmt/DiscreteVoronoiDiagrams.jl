# Discrete Voronoï diagrams for Julia

[![Build Status](https://github.com/emmt/DiscreteVoronoiDiagrams.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/DiscreteVoronoiDiagrams.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/DiscreteVoronoiDiagrams.jl?svg=true)](https://ci.appveyor.com/project/emmt/DiscreteVoronoiDiagrams-jl) [![Coverage](https://codecov.io/gh/emmt/DiscreteVoronoiDiagrams.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/DiscreteVoronoiDiagrams.jl)

Package `DiscreteVoronoiDiagrams` computes `N`-dimensional discrete Voronoï
diagrams for Julia.

A discrete Voronoï diagram is an `N`-dimensional array, say `inds`, such that
`inds[i]` is the index of the node that is the nearest to the location `i`
among a given list of nodes.

Compared to
[`DiscreteVoronoi`](https://github.com/jacobusmmsmit/DiscreteVoronoi.jl), this
package implements only two strategies to build Voronoï diagrams (a slow
*reference* one and a faster one) but can deal with any number of dimensions
(not just 2) and nodes can be specified in two different ways (see below).

The `DiscreteVoronoiDiagrams` exports two methods: `discrete_voronoi` and
`discrete_voronoi!`, the latter being the in-place version of the former. There
are two possibilities to specify the centers of the Voronoï domains:

- To build a discrete Voronoï diagram of size `dims` for a list of nodes whose
  `N`-dimensional coordinates are the elements of the vector `A`, call:

  ``` julia
  discrete_voronoi(dims, A) -> inds
  ```

  The elements of `A` may be instances of `CardinalIndex{N}` or `N`-tuple of
  reals. The result `inds` is an array of (linear) indices such that:

  ``` julia
  i = inds[j]
  ```

  is the index of the node in `A` that is nearest to node `j` among all nodes
  in `A`. The coordinates of the `i`-th node in `A` is simply `A[i]` while the
  coordinates of the `j`-th node in the Voronoï diagram are its Cartesian
  coordinates in the array `inds` representing the diagram. If `A` is empty,
  the result is an array filled with `firstindex(A) - 1`.

- To build a discrete Voronoï diagram for the nodes marked as `true` in a
  Boolean array `B`, call:

  ``` julia
  discrete_voronoi(B) -> inds
  ```

  The result `inds` is an array of (linear) indices of the nearest marked nodes
  for each node of `B`. In other words:

  ``` julia
  i = inds[j]
  ```

  is the index of the marked node in `B` (such that `B[i] == true` holds) that
  is the nearest to node `j` among all marked nodes in `B`. If there are no
  marked nodes in `B`, the result is an array filled with `firstindex(B) - 1`.

Keyword `dist` may be used to specify another distance than the (squared)
Euclidean distance to build the diagram. In case of tie, the first in the
lexicographic order of `A` (resp. `B`) is favored.
