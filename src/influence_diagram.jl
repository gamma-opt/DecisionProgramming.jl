using Base.Iterators: product


# --- Nodes and States ---

"""Node type. Alias for `Int`."""
const Node = Int

function validate_node(j::Node, I_j::Vector{Node})
    if !allunique(I_j)
        throw(DomainError("All information nodes should be unique."))
    end
    if !all(i < j for i in I_j)
        throw(DomainError("All nodes in the information set must be less than node j."))
    end
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
    function States(vals::Vector{State})
        if !all(vals .≥ 1)
            throw(DomainError("All states must be ≥ 1."))
        end
        new(vals)
    end
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
    if length(S) != n
        throw(DomainError("Each change and decision node should have states."))
    end
    if Set(c.j for c in C) ∪ Set(d.j for d in D) != Set(1:n)
        throw(DomainError("Union of change and decision nodes should be {1,...,n}."))
    end
    if Set(v.j for v in V) != Set((n+1):(n+length(V)))
        throw(DomainError("Values nodes should be {n+1,...,n+|V|}."))
    end
    I_V = union((v.I_j for v in V)...)
    if !(I_V ⊆ Set(1:n))
        throw(DomainError("Each information set I(v) for value node v should be a subset of C∪D."))
    end
    # Check for redundant nodes.
    leaf_nodes = setdiff(1:n, (c.I_j for c in C)..., (d.I_j for d in D)...)
    for i in leaf_nodes
        if !(i∈I_V)
            @warn("Chance or decision node $i is redundant.")
        end
    end
    for v in V
        if isempty(v.I_j)
            @warn("Value node $(v.j) is redundant.")
        end
    end
end


# --- Paths ---

"""Path type. Alias for `NTuple{N, State} where N`."""
const Path{N} = NTuple{N, State} where N

"""Iterate over paths in lexicographical order.

# Examples
```julia-repl
julia> states = States([2, 3])
julia> vec(collect(paths(states)))
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
julia> vec(collect(paths(states, Dict(1=>2))))
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

"""ForbiddenPath type.

# Examples
```julia
ForbiddenPath[
    ([1, 2], Set([(1, 2)])),
    ([3, 4, 5], Set([(1, 2, 3), (3, 4, 5)]))
]
```
"""
const ForbiddenPath = Tuple{Vector{Node}, Set{Path}}


# --- Probabilities ---

"""Construct and validate stage probabilities.

# Examples
```julia-repl
julia> data = [0.5 0.5 ; 0.2 0.8]
julia> X = Probabilities(2, data)
julia> s = (1, 2)
julia> X(s)
0.5
```
"""
struct Probabilities{N} <: AbstractArray{Float64, N}
    j::Node
    data::Array{Float64, N}
    function Probabilities(j::Node, data::Array{Float64, N}) where N
        if !all(x > 0 for x in data)
            @warn("The influence diagram contains inactive chance states. Do not use active paths cut.")
        end
        for i in CartesianIndices(size(data)[1:end-1])
            if !(sum(data[i, :]) ≈ 1)
                throw(DomainError("Probabilities should sum to one."))
            end
        end
        new{N}(j, data)
    end
end

Base.size(P::Probabilities) = size(P.data)
Base.IndexStyle(::Type{<:Probabilities}) = IndexLinear()
Base.getindex(P::Probabilities, i::Int) = getindex(P.data, i)
Base.getindex(P::Probabilities, I::Vararg{Int,N}) where N = getindex(P.data, I...)

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
s = (1, 2)
P(s)
```
"""
struct DefaultPathProbability <: AbstractPathProbability
    C::Vector{ChanceNode}
    X::Vector{Probabilities}
end

function (P::DefaultPathProbability)(s::Path)
    prod(X(s[[c.I_j; c.j]]) for (c, X) in zip(P.C, P.X))
end


# --- Consequences ---

"""State utilities.

# Examples
```julia-repl
julia> vals = [1.0 -2.0; 3.0 4.0]
julia> Y = Consequences(3, vals)
julia> s = (1, 2)
julia> Y(s)
-2.0
```
"""
struct Consequences{N} <: AbstractArray{Float64, N}
    j::Node
    data::Array{Float64, N}
end

Base.size(Y::Consequences) = size(Y.data)
Base.IndexStyle(::Type{<:Consequences}) = IndexLinear()
Base.getindex(Y::Consequences, i::Int) = getindex(Y.data, i)
Base.getindex(Y::Consequences, I::Vararg{Int,N}) where N = getindex(Y.data, I...)

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

"""Default path utility.

# Examples
```julia
U = DefaultPathUtility(V, Y)
s = (1, 2)
U(s)
```
"""
struct DefaultPathUtility <: AbstractPathUtility
    V::Vector{ValueNode}
    Y::Vector{Consequences}
end

function (U::DefaultPathUtility)(s::Path)
    sum(Y(s[v.I_j]) for (v, Y) in zip(U.V, U.Y))
end


# --- Local Decision Strategy ---

"""Local decision strategy type.

# Examples
```julia
Z = LocalDecisionStrategy(1, data)
Z(s_I)
```
"""
struct LocalDecisionStrategy{N} <: AbstractArray{Int, N}
    j::Node
    data::Array{Int, N}
    function LocalDecisionStrategy(j::Node, data::Array{Int, N}) where N
        if !all(0 ≤ x ≤ 1 for x in data)
            throw(DomainError("All values x must be 0 ≤ x ≤ 1."))
        end
        for s_I in CartesianIndices(size(data)[1:end-1])
            if !(sum(data[s_I, :]) == 1)
                throw(DomainError("Values should add to one."))
            end
        end
        new{N}(j, data)
    end
end

Base.size(Z::LocalDecisionStrategy) = size(Z.data)
Base.IndexStyle(::Type{<:LocalDecisionStrategy}) = IndexLinear()
Base.getindex(Z::LocalDecisionStrategy, i::Int) = getindex(Z.data, i)
Base.getindex(Z::LocalDecisionStrategy, I::Vararg{Int,N}) where N = getindex(Z.data, I...)

function (Z::LocalDecisionStrategy)(s_I::Path)::State
    findmax(Z[s_I..., :])[2]
end


# --- Decision Strategy ---

"""Decision strategy type."""
struct DecisionStrategy
    D::Vector{DecisionNode}
    Z_j::Vector{LocalDecisionStrategy}
end
