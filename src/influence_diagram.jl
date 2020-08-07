using Parameters
using Base.Iterators: product

"""Node type alias."""
const Node = Int

function validate_node(j::Node, I_j::Vector{Node})
    I_j = sort(unique(I_j))
    all(i < j for i in I_j) || error("All nodes in the information set must be less than node j.")
    return j, I_j
end

"""Chance node type."""
struct ChanceNode
    j::Node
    I_j::Vector{Node}
    ChanceNode(j, I_j) = new(validate_node(j, I_j)...)
end

"""Decision node type."""
struct DecisionNode
    j::Node
    I_j::Vector{Node}
    DecisionNode(j, I_j) = new(validate_node(j, I_j)...)
end

"""Value node type."""
struct ValueNode
    j::Node
    I_j::Vector{Node}
    ValueNode(j, I_j) = new(validate_node(j, I_j)...)
end

"""State type alias."""
const State = Int

"""States type."""
struct States <: AbstractArray{State, 1}
    vals::Vector{State}
    States(vals) = all(vals .≥ 1) ? new(vals) : error("All states must be ≥ 1.")
end

Base.size(S::States) = size(S.vals)
Base.IndexStyle(::Type{<:States}) = IndexLinear()
Base.getindex(S::States, i::Int) = getindex(S.vals, i)
Base.length(S::States) = length(S.vals)
Base.eltype(S::States) = eltype(S.vals)

"""Path type alias."""
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

paths(S::States) = paths(S.vals)

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

paths(S::States, fixed::Dict{Int, Int}) = paths(S.vals, fixed)


# --- Probabilities ---

"""Construct and validate stage probabilities."""
struct Probabilities
    vals::Array{Float64, N} where N
    function Probabilities(X::Array{Float64, N} where N)
        all(x > 0 for x in X) || @warn("Probabilities are not all positive, do not use number of paths cuts.")
        # TODO: indexing without paths function
        states = Int[size(X)[1:end-1]...]
        for s_I in paths(states)
            sum(X[s_I..., :]) ≈ 1 || error("Probabilities should sum to one.")
        end
        new(X)
    end
end

"""Get probabilities of path `s`."""
Base.getindex(X::Probabilities, s::Path) = getindex(X.vals, s...)


# --- Consequences ---

"""State utilities."""
struct Consequences
    vals::Array{Float64, N} where N
end

"""Get consequences of path `s`."""
Base.getindex(Y::Consequences, s::Path) = getindex(Y.vals, s...)


# --- Path Probability ---

"""Path probability."""
struct PathProbability
    C::Vector{ChanceNode}
    X::Vector{Probabilities}
end

"""Evalute path probability."""
function (P::PathProbability)(s::Path)
    prod(X[s[[c.I_j; c.j]]] for (c, X) in zip(P.C, P.X))
end


# --- Path Utility ---

"""Abstract path utility type.

# Examples
```julia
struct PathUtility <: AbstractPathUtility
    V::Vector{ValueNode}
    Y::Vector{Consequences}
end

(U::PathUtility)(s::Path) = ...
```
"""
abstract type AbstractPathUtility end

"""Default path utility."""
struct DefaultPathUtility <: AbstractPathUtility
    V::Vector{ValueNode}
    Y::Vector{Consequences}
end

"""Evaluate default path utility."""
function (U::DefaultPathUtility)(s::Path)
    sum(Y[s[v.I_j]] for (v, Y) in zip(U.V, U.Y))
end


# --- Validate influence diagram

"""Validate influence diagram."""
function validate_influence_diagram(S::States, C::Vector{ChanceNode}, D::Vector{DecisionNode}, V::Vector{ValueNode}, X::Vector{Probabilities}, Y::Vector{Consequences})
    n = length(C) + length(D)
    N = n + length(V)
    C_j = [c.j for c in C]
    D_j = [d.j for d in D]
    V_j = [v.j for v in V]

    length(S) == n || error("Each change and decision node should have states.")

    # Validate nodes
    Set(C_j ∪ D_j) == Set(1:n) || error("Union of change and decision nodes should be {1,...,n}.")
    Set(V_j) == Set((n+1):N) || error("Values nodes should be {n+1,...,n+|V|}.")

    # TODO: avoid implicit sorting
    k1 = sortperm(C_j)
    k2 = sortperm(D_j)
    k3 = sortperm(V_j)

    return S, C[k1], D[k2], V[k3], X[k1], Y[k3]
end
