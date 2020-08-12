using Parameters
using Base.Iterators: product


# --- Nodes and States ---

"""Node type. Alias for `Int`."""
const Node = Int

function validate_node(j::Node, I_j::Vector{Node})
    I_j = sort(unique(I_j))
    all(i < j for i in I_j) || error("All nodes in the information set must be less than node j.")
    return j, I_j
end

"""Chance node type.

# Examples
```julia
c = ChanceNode(3, [1, 2])
```
"""
struct ChanceNode
    j::Node
    I_j::Vector{Node}
    function ChanceNode(j::Node, I_j::Vector{Node})
        j, I_j = validate_node(j, I_j)
        new(j, I_j)
    end
end

"""Decision node type.

# Examples
```julia
d = DecisionNode(2, [1])
```
"""
struct DecisionNode
    j::Node
    I_j::Vector{Node}
    function DecisionNode(j::Node, I_j::Vector{Node})
        j, I_j = validate_node(j, I_j)
        new(j, I_j)
    end
end

"""Value node type.

# Examples
```julia
v = ValueNode(4, [1, 3])
```
"""
struct ValueNode
    j::Node
    I_j::Vector{Node}
    function ValueNode(j::Node, I_j::Vector{Node})
        j, I_j = validate_node(j, I_j)
        new(j, I_j)
    end
end

"""State type. Alias for `Int`."""
const State = Int

"""States type. Works like `Vector{State}`.

# Examples
```julia
S = States([2, 3, 2, 4])
```
"""
struct States <: AbstractArray{State, 1}
    vals::Vector{State}
    States(vals::Vector{State}) = all(vals .≥ 1) ? new(vals) : error("All states must be ≥ 1.")
end

Base.size(S::States) = size(S.vals)
Base.IndexStyle(::Type{<:States}) = IndexLinear()
Base.getindex(S::States, i::Int) = getindex(S.vals, i)
Base.length(S::States) = length(S.vals)
Base.eltype(S::States) = eltype(S.vals)

"""Construct states from vector of (state, nodes) tuples.

# Examples
```julia-repl
julia> S = States([(2, [1, 3]), (3, [2, 4, 5])])
States([2, 3, 2, 3, 3])
```
"""
function States(states::Vector{Tuple{State, Vector{Node}}})
    S_j = Vector{State}(undef, sum(length(j) for (_, j) in states))
    for (s, j) in states
        S_j[j] .= s
    end
    States(S_j)
end

"""Validate influence diagram."""
function validate_influence_diagram(S::States, C::Vector{ChanceNode}, D::Vector{DecisionNode}, V::Vector{ValueNode})
    n = length(C) + length(D)
    length(S) == n || error(
        "Each change and decision node should have states.")
    Set(c.j for c in C) ∪ Set(d.j for d in D) == Set(1:n) || error(
        "Union of change and decision nodes should be {1,...,n}.")
    Set(v.j for v in V) == Set((n+1):(n+length(V))) || error(
        "Values nodes should be {n+1,...,n+|V|}.")
end


# --- Paths ---

"""Path type. Alias for `NTuple{N, State} where N`."""
const Path = NTuple{N, State} where N

"""Iterate over paths in lexicographical order.

# Examples
```julia-repl
julia> states = States([2, 3])
julia> collect(paths(states))[:]
[(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]
```
"""
function paths(states::AbstractVector{State})
    product(UnitRange.(one(eltype(states)), states)...)
end

"""Iterate over paths with fixed states in lexicographical order.

# Examples
```julia-repl
julia> states = States([2, 3])
julia> collect(paths(states, fixed=Dict(1=>2)))[:]
[(2, 1), (2, 2), (2, 3)]
```
"""
function paths(states::AbstractVector{State}, fixed::Dict{Int, Int})
    iters = collect(UnitRange.(one(eltype(states)), states))
    for (i, v) in fixed
        iters[i] = UnitRange(v, v)
    end
    product(iters...)
end


# --- Probabilities ---

"""Construct and validate stage probabilities.

# Examples
```julia
data = [0.5 0.5 ; 0.2 0.8]
X = Probabilities(data)
```
"""
struct Probabilities{N} <: AbstractArray{Float64, N}
    data::Array{Float64, N}
    function Probabilities(data::Array{Float64, N}) where N
        all(x > 0 for x in data) || @warn(
            "Probabilities are not all positive, do not use number of paths cuts.")
        for i in CartesianIndices(size(data)[1:end-1])
            sum(data[i, :]) ≈ 1 || error("Probabilities should sum to one.")
        end
        new{N}(data)
    end
end

Base.size(P::Probabilities) = size(P.data)
Base.IndexStyle(::Type{<:Probabilities}) = IndexLinear()
Base.getindex(P::Probabilities, i::Int) = getindex(P.data, i)
Base.getindex(P::Probabilities, I::Vararg{Int,N}) where N = getindex(P.data, I...)

"""Return probabilities of information path `s`.

# Examples
```julia-repl
julia> s = (1, 2)
julia> X(s)
0.5
```
"""
(X::Probabilities)(s::Path) = X[s...]


# --- Path Probability ---

"""Abstract path probability type.

# Examples
```julia
struct PathProbability <: AbstractPathProbability
    C::Vector{ChanceNode}
    # ...
end

(U::PathProbability)(s::Path) = ...
```
"""
abstract type AbstractPathProbability end

"""Path probability.

# Examples
```julia
P = DefaultPathProbability(C, X)
```
"""
struct DefaultPathProbability <: AbstractPathProbability
    C::Vector{ChanceNode}
    X::Vector{Probabilities}
end

"""Evalute path probability."""
function (P::DefaultPathProbability)(s::Path)
    prod(X(s[[c.I_j; c.j]]) for (c, X) in zip(P.C, P.X))
end


# --- Consequences ---

"""State utilities.

# Examples
```julia
vals = [1.0 -2.0; 3.0 4.0]
Y = Consequences(vals)
```
"""
struct Consequences{N} <: AbstractArray{Float64, N}
    data::Array{Float64, N}
end

Base.size(Y::Consequences) = size(Y.data)
Base.IndexStyle(::Type{<:Consequences}) = IndexLinear()
Base.getindex(Y::Consequences, i::Int) = getindex(Y.data, i)
Base.getindex(Y::Consequences, I::Vararg{Int,N}) where N = getindex(Y.data, I...)

"""Return consequences of information path `s`.

# Examples
```julia-repl
julia> s = (1, 2)
julia> Y(s)
-2.0
```
"""
(Y::Consequences)(s::Path) = Y[s...]


# --- Path Utility ---

"""Abstract path utility type.

# Examples
```julia
struct PathUtility <: AbstractPathUtility
    V::Vector{ValueNode}
    # ...
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
    sum(Y(s[v.I_j]) for (v, Y) in zip(U.V, U.Y))
end
