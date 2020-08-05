using Parameters
using Base.Iterators: product

# --- Influence Diagram ---

"""Node type."""
const Node = Int

"""State type."""
const State = Int

"""Influence diagram."""
struct InfluenceDiagram
    C::Vector{Node}
    D::Vector{Node}
    V::Vector{Node}
    A::Vector{Pair{Node, Node}}
    S_j::Vector{State}
    I_j::Vector{Vector{Node}}
end

"""Construct and validate an influence diagram.

# Examples
```julia
C = [1, 3]
D = [2, 4]
V = [5]
A = [1=>2, 3=>4, 2=>5, 3=>5]
S_j = [2, 3, 2, 4]
G = InfluenceDiagram(C, D, V, A, S_j)
```
"""
function InfluenceDiagram(C::Vector{Node}, D::Vector{Node}, V::Vector{Node}, A::Vector{Pair{Node, Node}}, S_j::Vector{State})
    # Enforce sorted and unique elements.
    C = sort(unique(C))
    D = sort(unique(D))
    V = sort(unique(V))
    A = sort(unique(A))

    # Sizes
    n = length(C) + length(D)
    N = n + length(V)

    ## Validate nodes
    Set(C ∪ D) == Set(1:n) || error("Union of change and decision nodes should be {1,...,n}.")
    Set(V) == Set((n+1):N) || error("Values nodes should be {n+1,...,n+|V|}.")

    ## Validate arcs
    # 1) Inclusion A ⊆ N×N.
    # 2) Graph is acyclic.
    # 3) There are no arcs from value nodes to other nodes.
    all(1 ≤ i < j ≤ N for (i, j) in A) || error("Forall (i,j)∈A we should have 1≤i<j≤N.")
    all(i∉V for (i, j) in A) || error("There should be no nodes from value nodes to other nodes.")

    ## Validate states
    # Each chance and decision node has a finite number of states
    length(S_j) == n || error("Each change and decision node should have states.")
    all(S_j[j] ≥ 1 for j in 1:n) || error("Each change and decision node should have ≥ 1 states.")

    # Construct the information set
    I_j = [Vector{Node}() for i in 1:N]
    for (i, j) in A
        push!(I_j[j], i)
    end

    InfluenceDiagram(C, D, V, A, S_j, I_j)
end


# --- Paths ---

"""Path type."""
const Path = NTuple{N, State} where N

"""Iterate over paths in lexicographical order.

# Examples
```julia-repl
julia> collect(paths([2, 3]))[:]
[(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]
```
"""
function paths(num_states::Vector{State})
    product(UnitRange.(one(eltype(num_states)), num_states)...)
end

"""Iterate over paths with fixed states in lexicographical order.

# Examples
```julia-repl
julia> collect(paths([2, 3], fixed=Dict(1=>2)))[:]
[(2, 1), (2, 2), (2, 3)]
```
"""
function paths(num_states::Vector{State}, fixed::Dict{Int, Int})
    iters = collect(UnitRange.(one(eltype(num_states)), num_states))
    for (i, v) in fixed
        iters[i] = UnitRange(v, v)
    end
    product(iters...)
end


# --- Probabilities ---

"""Type alias for probability."""
const Probability = Array{Float64, N} where N

"""Probabilities type."""
struct Probabilities
    X::Dict{Node, <:Probability}
end

"""Validate probabilities on an influence diagram.

# Examples
```julia
X = Probabilities(G, X)
```
"""
function Probabilities(G::InfluenceDiagram, X::Dict{Node, <:Probability})
    @unpack C, S_j, I_j = G
    for j in C
        size(X[j]) == Tuple(S_j[[I_j[j]; j]]) || error("Array should be dimension |S_I(j)|*|S_j|.")
        all(x > 0 for x in X[j]) || @warn("Probabilities are not all positive, do not use number of paths cuts.")
        for s_I in paths(S_j[I_j[j]])
            sum(X[j][[s_I...; s_j]...] for s_j in 1:S_j[j]) ≈ 1 || error("probabilities shoud sum to one.")
        end
    end
    Probabilities(X)
end

(X::Probabilities)(j::Node) = X.X[j]
(X::Probabilities)(j::Node, s::Path) = X.X[j][s...]
(X::Probabilities)(j::Node, s::Path, G::InfluenceDiagram) = X(j, s[[G.I_j[j]; j]])


# --- Consequences ---

"""Type alias for consequence."""
const Consequence = Array{Float64, N} where N

"""Consequences type."""
struct Consequences
    Y::Dict{Node, <:Consequence}
end

"""Validate consequences on an influence diagram.

# Examples
```julia
Y = Consequences(G, Y)
```
"""
function Consequences(G::InfluenceDiagram, Y::Dict{Node, <:Consequence})
    @unpack V, S_j, I_j = G
    for j in V
        size(Y[j]) == Tuple(S_j[I_j[j]]) || error("Array should be dimension |S_I(j)|.")
    end
    Consequences(Y)
end

(Y::Consequences)(j::Node) = Y.Y[j]
(Y::Consequences)(j::Node, s::Path) = Y.Y[j][s...]
(Y::Consequences)(j::Node, s::Path, G::InfluenceDiagram) = Y(j, s[G.I_j[j]])


# --- Path Probability ---

function path_probability(G::InfluenceDiagram, X::Probabilities, s::Path)
    prod(X(j, s, G) for j in G.C)
end

"""Path probability type.

# Examples
```julia
P = PathProbability(G, X)
```
"""
struct PathProbability
    G::InfluenceDiagram
    X::Probabilities
    min::Float64
    function PathProbability(G::InfluenceDiagram, X::Probabilities)
        x_min = minimum(path_probability(G, X, s) for s in paths(G.S_j))
        new(G, X, x_min)
    end
end

"""Evaluate path probability.

# Examples
```julia
P(s)
```
"""
(P::PathProbability)(s::Path) = path_probability(P.G, P.X, s)


# --- Path Utility ---

"""Path utility type.

# Examples
```julia
U = PathUtility(G, Y)
```
"""
struct PathUtility
    G::InfluenceDiagram
    Y::Consequences
end

"""Evaluate path utility. Can be overwritten.

# Examples
```julia
U(s)
```
"""
(U::PathUtility)(s::Path) = sum(U.Y(j, s, U.G) for j in U.G.V)


# --- Alternative Constructors ---

"""Chance node."""
struct ChanceNode
    j::Node
    I_j::Vector{Node}
    S_j::State
    X::Probability
end

"""Decision node."""
struct DecisionNode
    j::Node
    I_j::Vector{Node}
    S_j::State
end

"""Value node."""
struct ValueNode
    j::Node
    I_j::Vector{Node}
    Y::Consequence
end

"""Construct influence diagram from nodes."""
function InfluenceDiagram(C::Vector{ChanceNode}, D::Vector{DecisionNode}, V::Vector{ValueNode})
    C2 = [c.j for c in C]
    D2 = [d.j for d in D]
    V2 = [v.j for v in V]
    A = Vector{Pair{Node, Node}}()
    S_j = Vector{State}(undef, length(C) + length(D))
    for c in C
        append!(A, [i => c.j for i in c.I_j])
        S_j[c.j] = c.S_j
    end
    for d in D
        append!(A, [i => d.j for i in d.I_j])
        S_j[d.j] = d.S_j
    end
    for v in V
        append!(A, [i => v.j for i in v.I_j])
    end
    InfluenceDiagram(C2, D2, V2, A, S_j)
end

function Probabilities(G::InfluenceDiagram, C::Vector{ChanceNode})
    Probabilities(G, Dict(c.j => c.X for c in C))
end

function Consequences(G::InfluenceDiagram, V::Vector{ValueNode})
    Consequences(G, Dict(v.j => v.Y for v in V))
end
