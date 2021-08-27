using Base.Iterators: product


# --- Nodes and States ---

"""
    Node = Int

Primitive type for node index. Alias for `Int`.
"""
const Node = Int

"""
    Name = String

Primitive type for node names. Alias for `String`.
"""
const Name = String

"""
    abstract type AbstractNode end

Node type for directed, acyclic graph.
"""
abstract type AbstractNode end


struct ChanceNode <: AbstractNode
    name::Name
    I_j::Vector{Name}
    states::Vector{Name}
    function ChanceNode(name, I_j, states)
        return new(name, I_j, states)
    end

end

struct DecisionNode <: AbstractNode
    name::Name
    I_j::Vector{Name}
    states::Vector{Name}
    function DecisionNode(name, I_j, states)
        return new(name, I_j, states)
    end
end

struct ValueNode <: AbstractNode
    name::Name
    I_j::Vector{Name}
    function ValueNode(name, I_j)
        return new(name, I_j)
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
function States(states::Vector{Tuple{State, Vector{Node}}}) # TODO should this just be gotten rid of?
    S_j = Vector{State}(undef, sum(length(j) for (_, j) in states))
    for (s, j) in states
        S_j[j] .= s
    end
    States(S_j)
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
    c::Node
    data::Array{Float64, N}
    function Probabilities(c::Node, data::Array{Float64, N}) where N
        for i in CartesianIndices(size(data)[1:end-1])
            if !(sum(data[i, :]) ≈ 1)
                throw(DomainError("Probabilities should sum to one."))
            end
        end
        new{N}(c, data)
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
    C::Vector{Node}
    I_c::Vector{Vector{Node}}
    X::Vector{Probabilities}
    function DefaultPathProbability(C, I_c, X)
        if length(C) == length(I_c)
            new(C, I_c, X)
        else
            throw(DomainError("The number of chance nodes and information sets given to DefaultPathProbability should be equal."))
        end
    end

end

function (P::DefaultPathProbability)(s::Path)
    prod(X(s[[I_c; c]]) for (c, I_c, X) in zip(P.C, P.I_c, P.X))
end


# --- Utilities ---

"""
    Utility = Float32

Primitive type for utility. Alias for `Float32`.
"""
const Utility = Float32

"""
    Utilities{N} <: AbstractArray{Utility, N}

State utilities.

# Examples
```julia-repl
julia> vals = [1.0 -2.0; 3.0 4.0]
julia> Y = Utilities(3, vals)
julia> s = (1, 2)
julia> Y(s)
-2.0
```
"""
struct Utilities{N} <: AbstractArray{Utility, N}
    v::Node
    data::Array{Utility, N}
    function Utilities(v::Node, data::Array{Utility, N}) where N
        if any(isinf(u) for u in data)
            throw(DomainError("A value should be defined for each element of a utility matrix."))
        end
        new{N}(v, data)
    end
end

Base.size(Y::Utilities) = size(Y.data)
Base.IndexStyle(::Type{<:Utilities}) = IndexLinear()
Base.getindex(Y::Utilities, i::Int) = getindex(Y.data, i)
Base.getindex(Y::Utilities, I::Vararg{Int,N}) where N = getindex(Y.data, I...)

(Y::Utilities)(s::Path) = Y[s...]


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

t = -100.0
U(s, t)
```
"""
struct DefaultPathUtility <: AbstractPathUtility
    I_v::Vector{Vector{Node}}
    Y::Vector{Utilities}
end

function (U::DefaultPathUtility)(s::Path)
    sum(Y(s[I_v]) for (I_v, Y) in zip(U.I_v, U.Y))
end

function (U::DefaultPathUtility)(s::Path, t::Utility)
    U(s) + t
end

# --- Influence diagram ---


mutable struct InfluenceDiagram
    Nodes::Vector{AbstractNode}
    Names::Vector{Name}
    I_j::Vector{Vector{Node}}
    States::Vector{Vector{Name}}
    S::States
    C::Vector{Node}
    D::Vector{Node}
    V::Vector{Node}
    X::Vector{Probabilities}
    Y::Vector{Utilities}
    P::AbstractPathProbability
    U::AbstractPathUtility
    translation::Utility
    function InfluenceDiagram()
        new(Vector{AbstractNode}())
    end
end


# --- Adding nodes ---

function validate_node(diagram::InfluenceDiagram,
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

function add_node!(diagram::InfluenceDiagram, node::AbstractNode)
    if !isa(node, ValueNode)
        validate_node(diagram, node.name, node.I_j, states = node.states)
    else
        validate_node(diagram, node.name, node.I_j, value_node = true)
    end
    push!(diagram.Nodes, node)
end


# --- Adding Probabilities ---

struct ProbabilityMatrix{N} <: AbstractArray{Float64, N}
    nodes::Vector{Name}
    indices::Vector{Dict{Name, Int}}
    matrix::Array{Float64, N}
end

Base.size(PM::ProbabilityMatrix) = size(PM.matrix)
Base.getindex(PM::ProbabilityMatrix, I::Vararg{Int,N}) where N = getindex(PM.matrix, I...)
Base.setindex!(PM::ProbabilityMatrix, p::Float64, I::Vararg{Int,N}) where N = (PM.matrix[I...] = p)
Base.setindex!(PM::ProbabilityMatrix{N}, X, I::Vararg{Any, N}) where N = (PM.matrix[I...] .= X)


function ProbabilityMatrix(diagram::InfluenceDiagram, node::Name)
    if node ∉ diagram.Names
        throw(DomainError("Node $node should be added as a node to the influence diagram."))
    end
    if node ∉ diagram.Names[diagram.C]
        throw(DomainError("Only chance nodes can have probability matrices."))
    end

    # Find the node's indices and it's I_c nodes
    c = findfirst(x -> x==node, diagram.Names)
    nodes = [diagram.I_j[c]..., c]
    names = diagram.Names[nodes]

    indices = Vector{Dict{Name, Int}}()
    for j in nodes
        states = Dict{Name, Int}(state => i
            for (i, state) in enumerate(diagram.States[j])
        )
        push!(indices, states)
    end
    matrix = fill(0.0, diagram.S[nodes]...)

    return ProbabilityMatrix(names, indices, matrix)
end

function set_probability!(probability_matrix::ProbabilityMatrix, scenario::Array{String}, probability::Float64)
    index = Vector{Int}()
    for (i, s) in enumerate(scenario)
        if get(probability_matrix.indices[i], s, 0) == 0
            throw(DomainError("Node $(probability_matrix.nodes[i]) does not have a state called $s."))
        else
            push!(index, get(probability_matrix.indices[i], s, 0))
        end
    end

    probability_matrix[index...] = probability
end

function set_probability!(probability_matrix::ProbabilityMatrix, scenario::Array{Any}, probabilities::Array{Float64})
    index = Vector{Any}()
    for (i, s) in enumerate(scenario)
        if isa(s, Colon)
            push!(index, s)
        elseif get(probability_matrix.indices[i], s, 0) == 0
            throw(DomainError("Node $(probability_matrix.nodes[i]) does not have state $s."))
        else
            push!(index, get(probability_matrix.indices[i], s, 0))
        end
    end

    probability_matrix[index...] = probabilities
end


function add_probabilities!(diagram::InfluenceDiagram, node::Name, probabilities::AbstractArray{Float64, N}) where N
    c = findfirst(x -> x==node, diagram.Names)

    if c ∈ [j.c for j in diagram.X]
        throw(DomainError("Probabilities should be added only once for each node."))
    end

    if size(probabilities) == Tuple((diagram.S[j] for j in (diagram.I_j[c]..., c)))
        if isa(probabilities, ProbabilityMatrix)
            # Check that probabilities sum to one happesn in Probabilities
            push!(diagram.X, Probabilities(c, probabilities.matrix))
        else
            push!(diagram.X, Probabilities(c, probabilities))
        end
    else
        throw(DomainError("The dimensions of a probability matrix should match the node's states' and information states' cardinality. Expected $(Tuple((diagram.S[n] for n in (diagram.I_j[c]..., c)))) for node $name, got $(size(probabilities))."))
    end
end


# --- Adding Utilities ---

struct UtilityMatrix{N} <: AbstractArray{Utility, N}
    I_v::Vector{Name}
    indices::Vector{Dict{Name, Int}}
    matrix::Array{Utility, N}
end

Base.size(UM::UtilityMatrix) = size(UM.matrix)
Base.getindex(UM::UtilityMatrix, I::Vararg{Int,N}) where N = getindex(UM.matrix, I...)
Base.setindex!(UM::UtilityMatrix, y::T, I::Vararg{Int,N}) where {N, T<:Real} = (UM.matrix[I...] = y)
Base.setindex!(UM::UtilityMatrix{N}, Y, I::Vararg{Any, N}) where N = (UM.matrix[I...] .= Y)


function UtilityMatrix(diagram::InfluenceDiagram, node::Name)
    if node ∉ diagram.Names
        throw(DomainError("Node $node should be added as a node to the influence diagram."))
    end
    if node ∉ diagram.Names[diagram.V]
        throw(DomainError("Only value nodes can have consequence matrices."))
    end

    # Find the node's indexand it's I_v nodes
    v = findfirst(x -> x==node, diagram.Names)
    I_v = diagram.I_j[v]
    names = diagram.Names[I_v]

    indices = Vector{Dict{Name, Int}}()
    for j in I_v
        states = Dict{Name, Int}(state => i
            for (i, state) in enumerate(diagram.States[j])
        )
        push!(indices, states)
    end
    matrix = Array{Utility}(fill(Inf, diagram.S[I_v]...))

    return UtilityMatrix(names, indices, matrix)
end

function set_utility!(utility_matrix::UtilityMatrix, scenario::Array{String}, utility::Real)
    index = Vector{Int}()
    for (i, s) in enumerate(scenario)
        if get(utility_matrix.indices[i], s, 0) == 0
            throw(DomainError("Node $(utility_matrix.I_v[i]) does not have a state called $s."))
        else
            push!(index, get(utility_matrix.indices[i], s, 0))
        end
    end

    utility_matrix[index...] = utility
end

function set_utility!(utility_matrix::UtilityMatrix, scenario::Array{Any}, utility::Array{T}) where T<:Real
    index = Vector{Any}()
    for (i, s) in enumerate(scenario)
        if isa(s, Colon)
            push!(index, s)
        elseif get(utility_matrix.indices[i], s, 0) == 0
            throw(DomainError("Node $(utility_matrix.I_v[i]) does not have state $s."))
        else
            push!(index, get(utility_matrix.indices[i], s, 0))
        end
    end

    utility_matrix[index...] = utility
end

function add_utilities!(diagram::InfluenceDiagram, node::Name, utilities::AbstractArray{T, N}) where {N,T<:Real}
    v = findfirst(x -> x==node, diagram.Names)

    if v ∈ [j.v for j in diagram.Y]
        throw(DomainError("Utilities should be added only once for each node."))
    end

    if size(utilities) == Tuple((diagram.S[j] for j in diagram.I_j[v]))
        if isa(utilities, UtilityMatrix)
            push!(diagram.Y, Utilities(Node(v), utilities.matrix))
        else
            # Conversion to Float32 using Utility(), since machine default is Float64
            push!(diagram.Y, Utilities(Node(v), [Utility(u) for u in utilities]))
        end
    else
        throw(DomainError("The dimensions of the utilities matrix should match the node's information states' cardinality. Expected $(Tuple((diagram.S[n] for n in diagram.I_j[v]))) for node $name, got $(size(utilities))."))
    end
end


# --- Generating Arcs ---

function validate_structure(Nodes::Vector{AbstractNode}, C_and_D::Vector{AbstractNode}, n_CD::Int, V::Vector{AbstractNode}, n_V::Int)
    # Validating node structure
    if n_CD == 0
        throw(DomainError("The influence diagram must have chance or decision nodes."))
    end
    if !(union((n.I_j for n in Nodes)...) ⊊ Set(n.name for n in Nodes))
        throw(DomainError("Each node that is part of an information set should be added as a node."))
    end
    # Checking the information sets of C and D nodes
    if !isempty(union((j.I_j for j in C_and_D)...) ∩ Set(v.name for v in V))
        throw(DomainError("Information sets should not include any value nodes."))
    end
    # Checking the information sets of V nodes
    if !isempty(V) && !isempty(union((v.I_j for v in V)...) ∩ Set(v.name for v in V))
        throw(DomainError("Information sets should not include any value nodes."))
    end
    # Check for redundant chance or decision nodes.
    last_CD_nodes = setdiff((j.name for j in C_and_D), (j.I_j for j in C_and_D)...)
    for i in last_CD_nodes
        if !isempty(V) && i ∉ union((v.I_j for v in V)...)
            @warn("Node $i is redundant.")
        end
    end
end


function generate_arcs!(diagram::InfluenceDiagram)

    # Chance and decision nodes
    C_and_D = filter(x -> !isa(x, ValueNode), diagram.Nodes)
    n_CD = length(C_and_D)
    # Value nodes
    V_nodes = filter(x -> isa(x, ValueNode), diagram.Nodes)
    n_V = length(V_nodes)

    validate_structure(diagram.Nodes, C_and_D, n_CD, V_nodes, n_V)

    # Declare vectors for results (final resting place InfluenceDiagram.Names and InfluenceDiagram.I_j)
    Names = Vector{Name}(undef, n_CD+n_V)
    I_j = Vector{Vector{Node}}(undef, n_CD+n_V)
    states = Vector{Vector{Name}}()
    S = Vector{State}(undef, n_CD)
    C = Vector{Node}()
    D = Vector{Node}()
    V = Vector{Node}()

    # Declare helper collections
    indices = Dict{Name, Node}()
    indexed_nodes = Set{Name}()
    # Declare index
    index = 1


    while true
        # Index nodes C and D that don't yet have indices but whose I_j have indices
        new_nodes = filter(j -> (j.name ∉ indexed_nodes && Set(j.I_j) ⊆ indexed_nodes), C_and_D)
        for j in new_nodes
            # Update helper collections
            push!(indices, j.name => index)
            push!(indexed_nodes, j.name)
            # Update results
            Names[index] = Name(j.name)    #TODO datatype conversion happens here, should we use push! ?
            I_j[index] = map(x -> Node(indices[x]), j.I_j)
            push!(states, j.states)
            S[index] = State(length(j.states))
            if isa(j, ChanceNode)
                push!(C, Node(index))
            else
                push!(D, Node(index))
            end
            # Increase index
            index += 1
        end

        # If no new nodes were indexed this iteration, terminate while loop
        if isempty(new_nodes)
            if index < n_CD
                throw(DomainError("The influence diagram should be acyclic."))
            else
                break
            end
        end
    end


    # Index value nodes
    for v in V_nodes
        # Update results
        Names[index] = Name(v.name)
        I_j[index] = map(x -> Node(indices[x]), v.I_j)
        push!(V, Node(index))
        # Increase index
        index += 1
    end

    diagram.Names = Names
    diagram.I_j = I_j
    diagram.States = states
    diagram.S = States(S)
    diagram.C = C
    diagram.D = D
    diagram.V = V
    # Declaring X and Y
    diagram.X = Vector{Probabilities}()
    diagram.Y = Vector{Utilities}()
end



# --- Generating Diagram ---

function generate_diagram!(diagram::InfluenceDiagram;
    default_probability::Bool=true,
    default_utility::Bool=true,
    positive_path_utility::Bool=false,
    negative_path_utility::Bool=false)


    # Sort probabilities and consequences
    sort!(diagram.X, by = x -> x.c)
    sort!(diagram.Y, by = x -> x.v)


    # Declare P and U if defaults are used
    if default_probability
        diagram.P = DefaultPathProbability(diagram.C, diagram.I_j[diagram.C], diagram.X)
    end
    if default_utility
        diagram.U = DefaultPathUtility(diagram.I_j[diagram.V], diagram.Y)
        if positive_path_utility
            # Conversion to Float32 using Utility(), since machine default is Float64
            diagram.translation = 1 -  minimum(diagram.U(s) for s in paths(diagram.S))
        elseif negative_path_utility
            diagram.translation = -1 - maximum(diagram.U(s) for s in paths(diagram.S))
        else
            diagram.translation = 0
        end
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
    d::Node
    data::Array{Int, N}
    function LocalDecisionStrategy(d::Node, data::Array{Int, N}) where N
        if !all(0 ≤ x ≤ 1 for x in data)
            throw(DomainError("All values x must be 0 ≤ x ≤ 1."))
        end
        for s_I in CartesianIndices(size(data)[1:end-1])
            if !(sum(data[s_I, :]) == 1)
                throw(DomainError("Values should add to one."))
            end
        end
        new{N}(d, data)
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
    D::Vector{Node}
    I_d::Vector{Vector{Node}}
    Z_d::Vector{LocalDecisionStrategy}
end
