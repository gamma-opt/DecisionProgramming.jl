using Base.Iterators: product


# --- Nodes and States ---

"""
    Node = Int

Primitive type for node. Alias for `Int`.
"""
const Node = Int

"""
    abstract type AbstractNode end

Node type for directed, acyclic graph.
"""
abstract type AbstractNode end

function validate_node(j::Node, I_j::Vector{Node})
    if !allunique(I_j)
        throw(DomainError("All information nodes should be unique."))
    end
    if !all(i < j for i in I_j)
        throw(DomainError("All nodes in the information set must be less than node j."))
    end
end

"""
    ChanceNode <: AbstractNode

Chance node type.

# Examples
```julia
c = ChanceNode(3, [1, 2])
```
"""
struct ChanceNode <: AbstractNode
    j::Node
    I_j::Vector{Node}
    function ChanceNode(j::Node, I_j::Vector{Node})
        validate_node(j, I_j)
        new(j, I_j)
    end
end

"""
    DecisionNode <: AbstractNode

Decision node type.

# Examples
```julia
d = DecisionNode(2, [1])
```
"""
struct DecisionNode <: AbstractNode
    j::Node
    I_j::Vector{Node}
    function DecisionNode(j::Node, I_j::Vector{Node})
        validate_node(j, I_j)
        new(j, I_j)
    end
end

"""
    ValueNode <: AbstractNode

Value node type.

# Examples
```julia
v = ValueNode(4, [1, 3])
```
"""
struct ValueNode <: AbstractNode
    j::Node
    I_j::Vector{Node}
    function ValueNode(j::Node, I_j::Vector{Node})
        validate_node(j, I_j)
        new(j, I_j)
    end
end

"""
    const State = Int

Primitive type for the number of states. Alias for `Int`.
"""
const State = Int


"""
    States <: AbstractArray{State, 1}

States type. Works like `Vector{State}`.

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

"""
    function States(states::Vector{Tuple{State, Vector{Node}}})

Construct states from vector of (state, nodes) tuples.

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


"""
    function validate_influence_diagram(S::States, C::Vector{ChanceNode}, D::Vector{DecisionNode}, V::Vector{ValueNode})

Validate influence diagram.
"""
function validate_influence_diagram(S::States, C::Vector{ChanceNode}, D::Vector{DecisionNode}, V::Vector{ValueNode})
    n = length(C) + length(D)
    # in validate_node_data
    if length(S) != n
        throw(DomainError("Each change and decision node should have states."))
    end

    # in deduce_node_indices logic...
    if Set(c.j for c in C) ∪ Set(d.j for d in D) != Set(1:n)
        throw(DomainError("Union of change and decision nodes should be {1,...,n}."))
    end

    # in deduce_node_indices logic...
    if Set(v.j for v in V) != Set((n+1):(n+length(V)))
        throw(DomainError("Values nodes should be {n+1,...,n+|V|}."))
    end

    # in deduce_node_indices
    I_V = union((v.I_j for v in V)...)
    if !(I_V ⊆ Set(1:n))
        throw(DomainError("Each information set I(v) for value node v should be a subset of C∪D."))
    end
    # in deduce_node_indices
    # Check for redundant nodes.
    leaf_nodes = setdiff(1:n, (c.I_j for c in C)..., (d.I_j for d in D)...)
    for i in leaf_nodes
        if !(i∈I_V)
            @warn("Chance or decision node $i is redundant.")
        end
    end
    # in validate_node_data
    for v in V
        if isempty(v.I_j)
            @warn("Value node $(v.j) is redundant.")
        end
    end
end


# --- Paths ---

"""
    const Path{N} = NTuple{N, State} where N

Path type. Alias for `NTuple{N, State} where N`.
"""
const Path{N} = NTuple{N, State} where N

"""
    function paths(states::AbstractVector{State})

Iterate over paths in lexicographical order.

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

"""
    function paths(states::AbstractVector{State}, fixed::Dict{Node, State})

Iterate over paths with fixed states in lexicographical order.

# Examples
```julia-repl
julia> states = States([2, 3])
julia> vec(collect(paths(states, Dict(1=>2))))
[(2, 1), (2, 2), (2, 3)]
```
"""
function paths(states::AbstractVector{State}, fixed::Dict{Node, State})
    iters = collect(UnitRange.(one(eltype(states)), states))
    for (i, v) in fixed
        iters[i] = UnitRange(v, v)
    end
    product(iters...)
end

"""
    const ForbiddenPath = Tuple{Vector{Node}, Set{Path}}

ForbiddenPath type.

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

"""
    struct Probabilities{N} <: AbstractArray{Float64, N}

Construct and validate stage probabilities.

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

"""
    abstract type AbstractPathProbability end

Abstract path probability type.

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

"""
    DefaultPathProbability <: AbstractPathProbability

Path probability.

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

"""
    Consequences{N} <: AbstractArray{Float64, N}

State utilities.

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

"""
    abstract type AbstractPathUtility end

Abstract path utility type.

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

"""
    DefaultPathUtility <: AbstractPathUtility

Default path utility.

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



# --- Influence diagram ---

"""
    Name = String

Primitive type for name of node. Alias for `String`.
"""
const Name = String


abstract type NodeData end

mutable struct InfluenceDiagram
    Nodes::Vector{NodeData}
    Names::Vector{Name}
    I_j::Vector{Vector{State}}
    S::States
    C::Vector{ChanceNode}
    D::Vector{DecisionNode}
    V::Vector{ValueNode}
    X::Vector{Probabilities}
    Y::Vector{Consequences}
    P::AbstractPathProbability
    U::AbstractPathUtility
    function InfluenceDiagram()
        new(Vector{NodeData}())
    end
end




# --- Node raw data ---

function validate_node_data(diagram::InfluenceDiagram,
    name::Name,
    I_j::Vector{Name};
    value_node::Bool=false,
    states::Vector{Name}=Vector{Name}())
    if !allunique([map(x -> x.name, diagram.Nodes)..., name])
        throw(DomainError("All node names should be unique."))
    end

    if !allunique(I_j)
        throw(DomainError("All nodes in an information set should be unique."))
    end

    if !allunique([name, I_j...])
        throw(DomainError("Node should not be included in its own information set."))
    end

    if !value_node
        if length(states) < 2
            throw(DomainError("Each chance and decision node should have more than one state."))
        end
    end

    if value_node
        if isempty(I_j)
            @warn("Value node $name is redundant.")
        end
    end
end

struct DecisionNodeData <: NodeData
    name::Name
    I_j::Vector{Name}
    states::Vector{Name}
    function DecisionNodeData(name::Name, I_j::Vector{Name}, states::Vector{Name})
        return new(name, I_j, states)
    end
end

function AddDecisionNode!(diagram::InfluenceDiagram,
    name::Name,
    I_j::Vector{Name},
    states::Vector{Name})
    validate_node_data(diagram, name, I_j, states = states)
    push!(diagram.Nodes, DecisionNodeData(name, I_j, states))
end

struct ChanceNodeData{N} <: NodeData
    name::Name
    I_j::Vector{Name}
    states::Vector{Name}
    probabilities::Array{Float64, N}
    function ChanceNodeData(name::Name, I_j::Vector{Name}, states::Vector{Name}, probabilities::Array{Float64, N}) where N
        return new{N}(name, I_j, states, probabilities)
    end

end

function AddChanceNode!(diagram::InfluenceDiagram,
    name::Name, I_j::Vector{Name},
    states::Vector{Name},
    probabilities::Array{Float64, N}) where N
    validate_node_data(diagram, name, I_j, states = states)
    push!(diagram.Nodes, ChanceNodeData(name, I_j, states, probabilities))
end

struct ValueNodeData{N} <: NodeData
    name::Name
    I_j::Vector{Name}
    consequences::Array{Float64, N}
    function ValueNodeData(name::Name, I_j::Vector{Name}, consequences::Array{Float64, N}) where N
        return new{N}(name, I_j, consequences)
    end
end

function AddValueNode!(diagram::InfluenceDiagram,
    name::Name,
    I_j::Vector{Name},
    consequences::Array{Float64, N}) where N
    validate_node_data(diagram, name, I_j, value_node = true)
    push!(diagram.Nodes, ValueNodeData(name, I_j, Array{Float64}(consequences)))
end


function deduce_node_indices(Nodes::Vector{NodeData})

    # Chance and decision nodes
    C_and_D = filter(x -> !isa(x, ValueNodeData), Nodes)
    n_CD = length(C_and_D)

    # Value nodes
    V = filter(x -> isa(x, ValueNodeData), Nodes)
    n_V = length(V)


    # Validating node structure
    if !isempty(Set(j.I_j for j in C_and_D) ∩ Set(v.name for v in V))
        throw(DomainError("Information set I(j) for chance or decision node j should not include value nodes."))
    end

    if !isempty(Set(v.I_j for v in V) ∩ Set(v.name for v in V))
        throw(DomainError("Information set I(v) for value nodes v should not include value nodes."))
    end
    # Check for redundant chance or decision nodes.
    last_CD_nodes = setdiff((j.name for j in C_and_D), (j.I_j for j in C_and_D)...)
    for i in last_CD_nodes
        if i ∉ union((v.I_j for v in V)...)
            @warn("Chance or decision node $i is redundant.")
        end
    end


    # Declare vectors for results (final resting place InfluenceDiagram.Names and InfluenceDiagram.I_j)
    Names = Vector{Name}(undef, n_CD+n_V)
    I_js = Vector{Vector{Node}}(undef, n_CD+n_V)

    # Declare helper collections
    indices = Dict{Name, Node}()
    indexed_nodes = Set{Name}()

    # Declare index for indexing nodes and loop iteration counter k
    index = 1
    k = 1

    while index <= n_CD+n_V && k == index

        # First index nodes C and D nodes
        if index <= n_CD
            for j in C_and_D
                # Give bindex if node does not already have an index and all of its I_j have an index
                if j.name ∉ indexed_nodes && Set(j.I_j) ⊆ indexed_nodes
                    # Update helper collections
                    push!(indices, j.name => index)
                    push!(indexed_nodes, j.name)

                    # Update results
                    Names[index] = j.name
                    I_js[index] = map(x -> indices[x], j.I_j)

                    # Increase index
                    index += 1
                end
            end

        # After indexing all C and D nodes, index value nodes
        else

            for v in V
                # Update results
                Names[index] = v.name
                I_js[index] = map(x -> indices[x], v.I_j)

                # Increase index
                index += 1
            end
        end

        # If new nodes have not been indexed during this iteration, terminate while loop
        if index <= k
            k += 1
        else
            k = index
        end
    end


    if index != k
        throw(DomainError("The influence diagram should be acyclic."))
    end

    return Names, I_js
end


function GenerateDiagram!(diagram::InfluenceDiagram; default_probability::Bool=true, default_utility::Bool=true)

    # Number of nodes
    nodes = [(n.name for n in diagram.Nodes)...]
    n = length(nodes)

    # Check all information sets are subsets of all nodes
    information_sets = union((n.I_j for n in diagram.Nodes)...)
    if !all(information_sets ⊊ nodes)
        throw(DomainError("Each node that is part of an information set should be added as a node."))
    end

    # Deduce indices for nodes
    diagram.Names, diagram.I_j = deduce_node_indices(diagram.Nodes)

    # Declare states, C, D, V, X, Y
    states = Vector{State}()
    diagram.C = Vector{ChanceNode}()
    diagram.D = Vector{DecisionNode}()
    diagram.V = Vector{ValueNode}()
    diagram.X = Vector{Probabilities}()
    diagram.Y = Vector{Consequences}()

    for (j, name) in enumerate(diagram.Names)
        node = diagram.Nodes[findfirst(x -> x.name == diagram.Names[j], diagram.Nodes)]

        if isa(node, DecisionNodeData)
            push!(states, length(node.states))
            push!(diagram.D, DecisionNode(j, diagram.I_j[j]))

        elseif isa(node, ChanceNodeData)
            push!(states, length(node.states))
            push!(diagram.C, ChanceNode(j, diagram.I_j[j]))

            # Check dimensions of probabiltiies match states of (I_j, j)
            if size(node.probabilities) == Tuple((states[n] for n in (diagram.I_j[j]..., j)))
                push!(diagram.X, Probabilities(j, node.probabilities))
            else
                # TODO rephrase this error message
                throw(DomainError("The dimensions of the probability matrix of node $name should match the number of states its information set and it has. In this case ", Tuple((states[n] for n in (diagram.I_j[j]..., j))), "."))
            end
        elseif isa(node, ValueNodeData)
            push!(diagram.V, ValueNode(j, diagram.I_j[j]))

            # Check dimensions of consequences match states of I_j
            if size(node.consequences) == Tuple((states[n] for n in diagram.I_j[j]))
                push!(diagram.Y, Consequences(j, node.consequences))
            else
                # TODO rephrase this error message
                throw(DomainError("The dimensions of the consequences matrix of node $name should match the number of states its information set has. In this case ", Tuple((states[n] for n in (diagram.I_j[j]..., j))), "."))
            end

        end
    end
    # TODO ask Olli about this, if the States should be changed
    diagram.S = States(states)

    # Validate influence diagram
    validate_influence_diagram(diagram.S, diagram.C, diagram.D, diagram.V)
    sort!.((diagram.C, diagram.D, diagram.V, diagram.X, diagram.Y), by = x -> x.j)

    # Declare P and U if defaults are used
    if default_probability
        diagram.P = DefaultPathProbability(diagram.C, diagram.X)
    end
    if default_utility
        diagram.U = DefaultPathUtility(diagram.V, diagram.Y)
    end

end

# --- Local Decision Strategy ---

"""
    LocalDecisionStrategy{N} <: AbstractArray{Int, N}

Local decision strategy type.

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

"""
    DecisionStrategy

Decision strategy type.
"""
struct DecisionStrategy
    D::Vector{DecisionNode}
    Z_j::Vector{LocalDecisionStrategy}
end
