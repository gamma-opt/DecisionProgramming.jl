using Parameters
using Base.Iterators: product

"""Node type alias."""
const Node = Int

"""Chance node."""
struct ChanceNode
    j::Node
    I_j::Vector{Node}
end

struct DecisionNode
    j::Node
    I_j::Vector{Node}
end

"""Value node."""
struct ValueNode
    j::Node
    I_j::Vector{Node}
end

"""State type alias."""
const State = Int

"""States type."""
struct States <: AbstractArray{State, 1}
    values::Vector{State}
    # TODO: validate state ≥ 1
end

Base.size(S::States) = size(S.values)
Base.IndexStyle(::Type{<:States}) = IndexLinear()
Base.getindex(S::States, i::Int) = getindex(S.values, i)
# Base.getindex(S::States, I::Vararg{Int,N}) where N = getindex(S.values, I)
Base.length(S::States) = length(S.values)
Base.eltype(S::States) = eltype(S.values)

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

paths(S::States) = paths(S.values)

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

paths(S::States, fixed::Dict{Int, Int}) = paths(S.values, fixed)


# --- Probabilities ---

"""Construct and validate stage probabilities."""
struct Probabilities #<: AbstractArray{Float64, N} where N
    values::Array{Float64, N} where N
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

Base.getindex(X::Probabilities, s::Path) = getindex(X.values, s...)
# Base.getindex(X::Probabilities, i::Int) = getindex(X.values, i)


# --- Consequences ---

"""State utilities."""
struct Consequences #<: AbstractArray{Float64, N} where N
    values::Array{Float64, N} where N
end

Base.getindex(Y::Consequences, s::Path) = getindex(Y.values, s...)
# Base.getindex(Y::Consequences, i::Int) = getindex(Y.values, i)


# --- Path Probability ---

struct PathProbability
    C::Vector{ChanceNode}
    X::Vector{Probabilities}
end

function (P::PathProbability)(s::Path)
    prod(X[s[[c.I_j; c.j]]] for (c, X) in zip(P.C, P.X))
end


# --- Path Utility ---

abstract type AbstractPathUtility end

struct DefaultPathUtility <: AbstractPathUtility
    V::Vector{ValueNode}
    Y::Vector{Consequences}
end

function (U::DefaultPathUtility)(s::Path)
    sum(Y[s[v.I_j]] for (v, Y) in zip(U.V, U.Y))
end
