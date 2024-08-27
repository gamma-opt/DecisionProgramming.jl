using Base.Iterators: product
using Random
using DataStructures

# --- Nodes and States ---

"""
    Node = Int16

Primitive type for node index. Alias for `Int16`.
"""
const Node = Int16

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

Base.show(io::IO, node::AbstractNode) = begin
    print_node_io(io, node)
end


"""
    struct ChanceNode <: AbstractNode

A struct for chance nodes, includes the name, information set, states and index of the node
"""
struct ChanceNode <: AbstractNode
    name::Name
    I_j::Vector{Name}
    states::Vector{Name}
    index::Node
    function ChanceNode(name, I_j, states)
        return new(name, I_j, states, 0)
    end
    function ChanceNode(name, I_j, states, index)
        return new(name, I_j, states, index)
    end
end

"""
    struct DecisionNode <: AbstractNode

A struct for decision nodes, includes the name, information set, states and index of the node
"""
struct DecisionNode <: AbstractNode
    name::Name
    I_j::Vector{Name}
    states::Vector{Name}
    index::Node
    function DecisionNode(name, I_j, states)
        return new(name, I_j, states, 0)
    end
    function DecisionNode(name, I_j, states, index)
        return new(name, I_j, states, index)
    end
end

"""
    struct ValueNode <: AbstractNode

A struct for value nodes, includes the name, information set and index of the node
"""
struct ValueNode <: AbstractNode
    name::Name
    I_j::Vector{Name}
    index::Node
    function ValueNode(name, I_j)
        return new(name, I_j, 0)
    end
    function ValueNode(name, I_j, index)
        return new(name, I_j, index)
    end
end



"""
    const State = Int16

Primitive type for the number of states. Alias for `Int16`.
"""
const State = Int16


"""
    struct States <: AbstractArray{State, 1}

States type. Works like `Vector{State}`.

# Examples
```julia-repl
julia> S = States(State.([2, 3, 2, 4]))
4-element States:
 2
 3
 2
 4
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



# --- Paths ---

"""
    const Path{N} = NTuple{N, State} where N

Path type. Alias for `NTuple{N, State} where N`.
"""
const Path{N} = NTuple{N, State} where N


"""
    const ForbiddenPath = Tuple{Vector{Node}, Set{Path}}

ForbiddenPath type.

# Examples
```julia-repl
julia> ForbiddenPath(([1, 2], Set([(1, 2)])))
(Int16[1, 2], Set(Tuple{Vararg{Int16,N}} where N[(1, 2)])

julia> ForbiddenPath[
    ([1, 2], Set([(1, 2)])),
    ([3, 4, 5], Set([(1, 2, 3), (3, 4, 5)]))
]
2-element Array{Tuple{Array{Int16,1},Set{Tuple{Vararg{Int16,N}} where N}},1}:
 ([1, 2], Set([(1, 2)]))
 ([3, 4, 5], Set([(1, 2, 3), (3, 4, 5)]))
```
"""
const ForbiddenPath = Tuple{Vector{Node}, Set{Path}}


"""
    const FixedPath = Dict{Node, State}

FixedPath type.

# Examples
```julia-repl
julia> FixedPath(Dict(1=>1, 2=>3))
Dict{Int16,Int16} with 2 entries:
  2 => 3
  1 => 1
```
"""
const FixedPath = Dict{Node, State}


"""
    function paths(states::AbstractVector{State})

Iterate over paths in lexicographical order.

# Examples
```julia-repl
julia> states = States(State.([2, 3]))
2-element States:
 2
 3

julia> vec(collect(paths(states)))
6-element Array{Tuple{Int16,Int16},1}:
 (1, 1)
 (2, 1)
 (1, 2)
 (2, 2)
 (1, 3)
 (2, 3)
```
"""
function paths(states::AbstractVector{State})
    product(UnitRange.(one(eltype(states)), states)...)
end

"""
    function paths(states::AbstractVector{State}, fixed::FixedPath)

Iterate over paths with fixed states in lexicographical order.

# Examples
```julia-repl
julia> states = States(State.([2, 3]))
2-element States:
 2
 3

julia> vec(collect(paths(states, Dict(Node(1) => State(2)))))
3-element Array{Tuple{Int16,Int16},1}:
 (2, 1)
 (2, 2)
 (2, 3)
```
"""
function paths(states::AbstractVector{State}, fixed::FixedPath)
    iters = collect(UnitRange.(one(eltype(states)), states))
    for (i, v) in fixed
        iters[i] = UnitRange(v, v)
    end
    product(iters...)
end


# --- Probabilities ---

"""
    struct Probabilities{N} <: AbstractArray{Float64, N}

Construct and validate stage probabilities (probabilities for a single node).

# Examples
```julia-repl
julia> data = [0.5 0.5 ; 0.2 0.8]
2×2 Array{Float64,2}:
 0.5  0.5
 0.2  0.8

julia> X = Probabilities(Node(2), data)
2×2 Probabilities{2}:
 0.5  0.5
 0.2  0.8

julia> s = (1, 2)
(1, 2)

julia> X(s)
0.5
```
"""
struct Probabilities{N} <: AbstractArray{Float64, N}
    data::Array{Float64, N}
    function Probabilities(data::Array{Float64, N}) where N
        for i in CartesianIndices(size(data)[1:end-1])
            if !(sum(data[i, :]) ≈ 1)
                throw(DomainError("Probabilities should sum to one."))
            end
        end
        new{N}(data)
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
"""
abstract type AbstractPathProbability end

"""
    struct DefaultPathProbability <: AbstractPathProbability

Path probability obtained as a product of the probability values corresponding to path s in each chance node.

# Examples
```julia-repl
julia> C = [2]
1-element Array{Int64,1}:
 2

julia> I_j = [[1]]
1-element Array{Array{Int64,1},1}:
 [1]

julia> X = [Probabilities(Node(2), [0.5 0.5; 0.2 0.8])]
1-element Array{Probabilities{2},1}:
 [0.5 0.5; 0.2 0.8]

julia> P = DefaultPathProbability(C, I_j, X)
DefaultPathProbability(Int16[2], Array{Int16,1}[[1]], Probabilities[[0.5 0.5; 0.2 0.8]])

julia> s = Path((1, 2))
(1, 2)

julia> P(s)
0.5
```
"""
struct DefaultPathProbability <: AbstractPathProbability
    C::Vector{Node}
    I_c::Vector{Vector{Node}}
    X::Vector{Probabilities}
    function DefaultPathProbability(C, I_c, X)
        if length(C) != length(I_c)
            throw(DomainError("The number of chance nodes and information sets given to DefaultPathProbability should be equal."))
        elseif length(C) != length(X)
            throw(DomainError("The number of chance nodes and probability matrices given to DefaultPathProbability should be equal."))
        else
            new(C, I_c, X)
        end
    end
end

function (P::DefaultPathProbability)(s::Path)
    prod(X(s[[I_c; c]]) for (c, I_c, X) in zip(P.C, P.I_c, P.X))
end


# --- Utilities ---

"""
    const Utility = Float32

Primitive type for utility. Alias for `Float32`.
"""
const Utility = Float32

"""
    struct Utilities{N} <: AbstractArray{Utility, N}

State utilities.

# Examples
```julia-repl
julia> vals = Utility.([1.0 -2.0; 3.0 4.0])
2×2 Array{Float32,2}:
 1.0  -2.0
 3.0   4.0

julia> Y = Utilities(Node(3), vals)
2×2 Utilities{2}:
 1.0  -2.0
 3.0   4.0

julia> s = Path((1, 2))
 (1, 2)

julia> Y(s)
-2.0f0
```
"""
struct Utilities{N} <: AbstractArray{Utility, N}
    data::Array{Utility, N}
    function Utilities(data::Array{Utility, N}) where N
        if any(isinf(u) for u in data)
            throw(DomainError("A value should be defined for each element of a utility matrix."))
        end
        new{N}(data)
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
"""
abstract type AbstractPathUtility end

"""
    struct DefaultPathUtility <: AbstractPathUtility

Default path utility obtained as a sum of the utility values corresponding to path s in each value node.

# Examples
```julia-repl
julia> vals = Utility.([1.0 -2.0; 3.0 4.0])
2×2 Array{Float32,2}:
 1.0  -2.0
 3.0   4.0

julia> Y = [Utilities(Node(3), vals)]
1-element Array{Utilities{2},1}:
 [1.0 -2.0; 3.0 4.0]

julia> I_3 = [[1,2]]
1-element Array{Array{Int64,1},1}:
 [1, 2]

julia> U = DefaultPathUtility(I_3, Y)
DefaultPathUtility(Array{Int16,1}[[1, 2]], Utilities[[1.0 -2.0; 3.0 4.0]])

julia> s = Path((1, 2))
(1, 2)

julia> U(s)
-2.0f0

julia> t = Utility(-100.0)


julia> U(s, t)
-102.0f0
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

"""
    struct RJT

A struct for rooted junction trees.
"""
struct RJT
    clusters::Dict{Name, Vector{Name}}
    arcs::Vector{Tuple{Name, Name}}
end


# --- Influence diagram ---
"""
    mutable struct InfluenceDiagram
        Nodes::OrderedDict{Name, AbstractNode}
        Names::Vector{Name}
        I_j::OrderedDict{Name, Vector{Name}}
        States::OrderedDict{Name, Vector{Name}}
        S::OrderedDict{Name,State}
    
        C::OrderedDict{Name, ChanceNode}
        D::OrderedDict{Name, DecisionNode}
        V::OrderedDict{Name, ValueNode}
        X::OrderedDict{Name, Probabilities}
        Y::OrderedDict{Name, Utilities}
        P::AbstractPathProbability
        U::AbstractPathUtility
        translation::Utility
        function InfluenceDiagram()
            new(OrderedDict{String, AbstractNode}())
        end
    end

Hold all information related to the influence diagram.

# Fields
All OrderedDicts are ordered by vector Names.
- `Nodes::OrderedDict{Name, AbstractNode}`: OrderedDict of node names as key
and their respective abstract nodes as values.
- `Names::Vector{Name}`: Names of nodes in order of their indices.
- `I_j::OrderedDict{Name, Vector{Name}}`: Information sets of nodes by their name.
- `States::OrderedDict{Name, Vector{Name}}`: States of each node by their name.
- `S::OrderedDict{Name,State}`: Number of states of each node.
- `C::OrderedDict{Name, ChanceNode}`: Chance nodes by their name.
- `D::OrderedDict{Name, DecisionNode}`: Decision nodes by their name.
- `V::OrderedDict{Name, ValueNode}`: Values nodes by their name.
- `X::OrderedDict{Name, Probabilities}`: Probability matrices of chance nodes by their name.
- `Y::OrderedDict{Name, Utilities}`: Utility matrices of value nodes by their name.
- `P::AbstractPathProbability`: Path probabilities.
- `U::AbstractPathUtility`: Path utilities.
- `translation::Utility`: Utility translation for storing the positive or negative
    utility translation.


# Examples
```julia
diagram = InfluenceDiagram()
```
"""
mutable struct InfluenceDiagram
    Nodes::OrderedDict{Name, AbstractNode}
    Names::Vector{Name}
    I_j::OrderedDict{Name, Vector{Name}}
    States::OrderedDict{Name, Vector{Name}}
    S::OrderedDict{Name,State}
 
    C::OrderedDict{Name, ChanceNode}
    D::OrderedDict{Name, DecisionNode}
    V::OrderedDict{Name, ValueNode}
    X::OrderedDict{Name, Probabilities}
    Y::OrderedDict{Name, Utilities}
    P::AbstractPathProbability
    U::AbstractPathUtility
    translation::Utility

    RJT::RJT
    function InfluenceDiagram()
        new(OrderedDict{Name, AbstractNode}())
    end
end


Base.show(io::IO, diagram::InfluenceDiagram) = begin
    println(io, "An influence diagram")
    println(io, "")
    println(io, "Node names:")
    println(io, diagram.Names)
    println(io, "")

    println(io, "Nodes:")
    println(io, "")

    for node in values(diagram.Nodes)
        print_node_io(io, node)
        println(io, "")
    end
end



# --- Adding nodes ---

function validate_node(diagram::InfluenceDiagram,
    name::Name,
    I_j::Vector{Name};
    value_node::Bool=false,
    states::Vector{Name}=Vector{Name}())

    if haskey(diagram.Nodes, name)
        throw(DomainError("All node names should be unique."))
    end

    if !allunique(I_j)
        throw(DomainError("All nodes in an information set should be unique."))
    end

    if name in I_j
        throw(DomainError("Node should not be included in its own information set."))
    end

    if value_node
        if isempty(I_j)
            @warn("Value node $name is redundant.")
        end
    end
end

"""
    function add_node!(diagram::InfluenceDiagram, node::AbstractNode)

Add node to influence diagram structure.

# Examples
```julia-repl
julia> add_node!(diagram, ChanceNode("O", [], ["lemon", "peach"]))
1-element Array{AbstractNode,1}:
 ChanceNode("O", String[], ["lemon", "peach"])
```
"""
function add_node!(diagram::InfluenceDiagram, node::AbstractNode)
    if !isa(node, ValueNode)
        validate_node(diagram, node.name, node.I_j, states = node.states)
    else
        validate_node(diagram, node.name, node.I_j, value_node = true)
    end
    diagram.Nodes[node.name] = node

end


# --- Adding Probabilities ---
"""
    struct ProbabilityMatrix{N} <: AbstractArray{Float64, N}
        nodes::Vector{Name}
        indices::Vector{Dict{Name, Int}}
        matrix::Array{Float64, N}
    end

Construct probability matrix.
"""
struct ProbabilityMatrix{N} <: AbstractArray{Float64, N}
    nodes::Vector{Name}
    indices::Vector{Dict{Name, Int}}
    matrix::Array{Float64, N}
end

Base.size(PM::ProbabilityMatrix) = size(PM.matrix)
Base.getindex(PM::ProbabilityMatrix, I::Vararg{Int,N}) where N = getindex(PM.matrix, I...)
function Base.setindex!(PM::ProbabilityMatrix, p::T, I::Vararg{Union{String, Int},N}) where {N, T<:Real}
    I2 = []
    for i in 1:N
        if isa(I[i], String)
            if get(PM.indices[i], I[i], 0) == 0
                throw(DomainError("Node $(PM.nodes[i]) does not have state $(I[i])."))
            end
            push!(I2, PM.indices[i][I[i]])
        else
            push!(I2, I[i])
        end
    end
    PM.matrix[I2...] = p
end
function Base.setindex!(PM::ProbabilityMatrix{N}, P::Array{T}, I::Vararg{Union{String, Int, Colon}, N}) where {N, T<:Real}
    I2 = []
    for i in 1:N
        if isa(I[i], Colon)
            push!(I2, :)
        elseif isa(I[i], String)
            if get(PM.indices[i], I[i], 0) == 0
                throw(DomainError("Node $(PM.nodes[i]) does not have state $(I[i])."))
            end
            push!(I2, PM.indices[i][I[i]])
        else
            push!(I2, I[i])
        end
    end
    PM.matrix[I2...] = P
end

"""
    function ProbabilityMatrix(diagram::InfluenceDiagram, node::Name)

Initialise a probability matrix for a given chance node. The matrix is initialised with zeros.

# Examples
```julia-repl
julia> X_O = ProbabilityMatrix(diagram, "O")
2-element ProbabilityMatrix{1}:
 0.0
 0.0
```
"""
function ProbabilityMatrix(diagram::InfluenceDiagram, node::Name)
    if !haskey(diagram.Nodes, node)
        throw(DomainError("Node $node should be added as a node to the influence diagram."))
    end
    if !haskey(diagram.C, node)
        throw(DomainError("Only chance nodes can have probability matrices."))
    end

    # Find the node's indices and it's I_c nodes
    names = [diagram.I_j[node]..., node]

    indices = [Dict{Name, Int}(state => i for (i, state) in enumerate(diagram.States[name])) for name in names]
    sizes = [diagram.S[name] for name in names]
    matrix = zeros(sizes...)

    return ProbabilityMatrix(names, indices, matrix)
end

"""
    function add_probabilities!(diagram::InfluenceDiagram, node::Name, probabilities::AbstractArray{Float64, N}) where N

Add probability matrix to influence diagram, specifically to its `X` vector.

# Examples
```julia
julia> X_O = ProbabilityMatrix(diagram, "O")
2-element ProbabilityMatrix{1}:
 0.0
 0.0

julia> X_O["lemon"] = 0.2
0.2

julia> add_probabilities!(diagram, "O", X_O)
ERROR: DomainError with Probabilities should sum to one.:

julia> X_O["peach"] = 0.8
0.2

julia> add_probabilities!(diagram, "O", X_O)
1-element Array{Probabilities,1}:
 [0.2, 0.8]
```
!!! note
    The function `generate_arcs!` must be called before probabilities or utilities can be added to the influence diagram.
"""
function add_probabilities!(diagram::InfluenceDiagram, node::Name, probabilities::AbstractArray{Float64, N}) where N
    if haskey(diagram.X, node)
        throw(DomainError("Probabilities should be added only once for each node."))
    end
    cardinalities = Tuple([diagram.S[n] for n in [diagram.I_j[node]..., node]])
    if size(probabilities) == cardinalities
        if isa(probabilities, ProbabilityMatrix)
            # Check that probabilities sum to one happens in Probabilities
            diagram.X[node] = Probabilities(probabilities.matrix)
        else
            diagram.X[node] = Probabilities(probabilities)
        end
    else
        throw(DomainError("The dimensions of a probability matrix should match the node's states' and information states' cardinality. Expected $cardinalities for node $node, got $(size(probabilities))."))
    end
end


# --- Adding Utilities ---

"""
    struct UtilityMatrix{N} <: AbstractArray{Utility, N}
        I_v::Vector{Name}
        indices::Vector{Dict{Name, Int}}
        matrix::Array{Utility, N}
    end

Construct utility matrix.
"""
struct UtilityMatrix{N} <: AbstractArray{Utility, N}
    I_v::Vector{Name}
    indices::Vector{Dict{Name, Int}}
    matrix::Array{Utility, N}
end

Base.size(UM::UtilityMatrix) = size(UM.matrix)
Base.getindex(UM::UtilityMatrix, I::Vararg{Int,N}) where N = getindex(UM.matrix, I...)
function Base.setindex!(UM::UtilityMatrix{N}, y::T, I::Vararg{Union{String, Int},N}) where {N, T<:Real}
    I2 = []
    for i in 1:N
        if isa(I[i], String)
            if get(UM.indices[i], I[i], 0) == 0
                throw(DomainError("Node $(UM.I_v[i]) does not have state $(I[i])."))
            end
            push!(I2, UM.indices[i][I[i]])
        else
            push!(I2, I[i])
        end
    end
    UM.matrix[I2...] = y
end
function Base.setindex!(UM::UtilityMatrix{N}, Y::Array{T}, I::Vararg{Union{String, Int, Colon}, N}) where {N, T<:Real}
    I2 = []
    for i in 1:N
        if isa(I[i], Colon)
            push!(I2, :)
        elseif isa(I[i], String)
            if get(UM.indices[i], I[i], 0) == 0
                throw(DomainError("Node $(UM.I_v[i]) does not have state $(I[i])."))
            end
            push!(I2, UM.indices[i][I[i]])
        else
            push!(I2, I[i])
        end
    end
    UM.matrix[I2...] = Y
end

"""
    function UtilityMatrix(diagram::InfluenceDiagram, node::Name)

Initialise a utility matrix for a value node. The matrix is initialised with `Inf` values.

# Examples
```julia-repl
julia> Y_V3 = UtilityMatrix(diagram, "V3")
2×3 UtilityMatrix{2}:
 Inf  Inf  Inf
 Inf  Inf  Inf
```
"""
function UtilityMatrix(diagram::InfluenceDiagram, node::Name)
    if !haskey(diagram.Nodes, node)
        throw(DomainError("Node $node should be added as a node to the influence diagram."))
    end
    if !haskey(diagram.V, node)
        throw(DomainError("Only value nodes can have consequence matrices."))
    end

    # Find the node's index and it's I_v nodes
    names = diagram.I_j[node]

    indices = [Dict{Name, Int}(state => i for (i, state) in enumerate(diagram.States[name])) for name in names]
    sizes = [diagram.S[name] for name in names]
    matrix = fill(Float32(Inf), sizes...)

    return UtilityMatrix(names, indices, matrix)
end



"""
    function add_utilities!(diagram::InfluenceDiagram, node::Name, utilities::AbstractArray{T, N}) where {N,T<:Real}

Add utility matrix to influence diagram, specifically to its `Y` vector.

# Examples
```julia-repl
julia> Y_V3 = UtilityMatrix(diagram, "V3")
2×3 UtilityMatrix{2}:
 Inf  Inf  Inf
 Inf  Inf  Inf

julia> Y_V3["peach", :] = [-40, -20, 0]
3-element Array{Int64,1}:
 -40
 -20
   0

julia> Y_V3["lemon", :] = [-200, 0, 0]
3-element Array{Int64,1}:
 -200
    0
    0

julia> add_utilities!(diagram, "V3", Y_V3)
1-element Array{Utilities,1}:
 [-200.0 0.0 0.0; -40.0 -20.0 0.0]

julia> add_utilities!(diagram, "V1", [0, -25])
2-element Array{Utilities,1}:
 [-200.0 0.0 0.0; -40.0 -20.0 0.0]
 [0.0, -25.0]
```
!!! note
    The function `generate_arcs!` must be called before probabilities or utilities can be added to the influence diagram.
"""
function add_utilities!(diagram::InfluenceDiagram, node::Name, utilities::AbstractArray{T, N}) where {N,T<:Real}
    if haskey(diagram.Y, node)
        throw(DomainError("Utilities should be added only once for each node."))
    end
    if any(u ==Inf for u in utilities)
        throw(DomainError("Utility values should be less than infinity."))
    end

    cardinalities = Tuple([diagram.S[n] for n in diagram.I_j[node]])
    if size(utilities) == cardinalities
        if isa(utilities, UtilityMatrix)
            diagram.Y[node] = Utilities(utilities.matrix)
        else
            # Conversion to Float32 using Utility(), since machine default is Float64
            diagram.Y[node] = Utilities([Utility(u) for u in utilities])
        end
    else
        throw(DomainError("The dimensions of the utilities matrix should match the node's information states' cardinality. Expected $cardinalities for node $node, got $(size(utilities))."))
    end
end


# --- Generating Arcs ---

function validate_structure(Nodes::OrderedDict{Name, AbstractNode}, C_and_D::OrderedDict{Name, AbstractNode}, n_CD::Int, V::OrderedDict{Name, AbstractNode}, n_V::Int)
    # Validating node structure
    if n_CD == 0
        throw(DomainError("The influence diagram must have at least one chance or decision node."))
    end
    if !(union((n.I_j for n in values(Nodes))...) ⊆ keys(Nodes))
        throw(DomainError("Each node that is part of an information set should be added as a node."))
    end
    # Checking the information sets
    if !isempty(union((j.I_j for j in values(Nodes))...) ∩ keys(V))
        throw(DomainError("Information sets should not include any value nodes."))
    end
    # Check for redundant chance or decision nodes.
    last_CD_nodes = setdiff(keys(C_and_D), union((j.I_j for j in values(C_and_D))...))
    for i in last_CD_nodes
        if !isempty(V) && i ∉ union((v.I_j for v in values(V))...)
            @warn("Node $i is redundant.")
        end
    end

    indexed_nodes = Set{Name}()
    while true
        new_nodes = filter(j -> (j ∉ indexed_nodes && Set(C_and_D[j].I_j) ⊆ indexed_nodes), keys(C_and_D))
        for j in new_nodes
            push!(indexed_nodes, j)
        end
        if isempty(new_nodes)
            if length(indexed_nodes) < n_CD
                throw(DomainError("The influence diagram should be acyclic."))
            else
                break
            end
        end
    end
end

"""
    function generate_arcs!(diagram::InfluenceDiagram)

Generate arc structures using nodes added to influence diagram and storing them to
variable diagram.I_j. Also creating variables diagram.States, diagram.S, diagram.C,
diagram.D and diagram.V and storing appropriate values to them.

# Examples
```julia
generate_arcs!(diagram)
```
"""
function generate_arcs!(diagram::InfluenceDiagram)
    C_and_D = filter(x -> !isa(x[2], ValueNode), pairs(diagram.Nodes)) # Collects all nodes not ValueNodes
    n_CD = length(C_and_D)
    V_ = filter(x -> isa(x[2], ValueNode), pairs(diagram.Nodes)) # Collects all ValueNodes
    n_V = length(V_)

    validate_structure(diagram.Nodes, C_and_D, n_CD, V_, n_V)

    # Declare vectors for results
    I_j = OrderedDict{Name, Vector{Name}}()
    states = OrderedDict{Name, Vector{Name}}()
    S = OrderedDict{Name, State}()
    C = OrderedDict{Name, ChanceNode}()
    D = OrderedDict{Name, DecisionNode}()
    V = OrderedDict{Name, ValueNode}()

    diagram.Nodes = merge(C_and_D, V_)

    #Assigning indices for all nodes (by constructing new nodes, because node-structs are immutable)
    node_index = 1
    for (name, node) in diagram.Nodes
        if isa(node, ChanceNode)
            diagram.Nodes[name] = ChanceNode(node.name, node.I_j, node.states, node_index)
        elseif isa(node, DecisionNode)
            diagram.Nodes[name] = DecisionNode(node.name, node.I_j, node.states, node_index)
        elseif isa(node, ValueNode)
            diagram.Nodes[name] = ValueNode(node.name, node.I_j, node_index)
        end
        node_index += 1
    end

    diagram.Names = get_keys(diagram.Nodes)

    for name in diagram.Names
        I_j[name] = diagram.Nodes[name].I_j
        if !isa(diagram.Nodes[name], ValueNode)
            states[name] = C_and_D[name].states
            S[name] = length(states[name])
        end
    end

    C = filter(x -> isa(x[2], ChanceNode), pairs(diagram.Nodes))
    D = filter(x -> isa(x[2], DecisionNode), pairs(diagram.Nodes))
    V = filter(x -> isa(x[2], ValueNode), pairs(diagram.Nodes))

    diagram.I_j = I_j
    diagram.States = states
    diagram.S = S
    diagram.C = C
    diagram.D = D
    diagram.V = V
    # Declaring X and Y
    diagram.X = OrderedDict{Name, Probabilities}()
    diagram.Y = OrderedDict{Name, Utilities}()
end

# --- Generating Diagram ---
"""
    function generate_diagram!(diagram::InfluenceDiagram;
    default_probability::Bool=true,
    default_utility::Bool=true,
    positive_path_utility::Bool=false,
    negative_path_utility::Bool=false)

Generate complete influence diagram with probabilities and utilities included.

# Arguments
- `default_probability::Bool=true`: Choice to use default path probabilities.
- `default_utility::Bool=true`: Choice to use default path utilities.
- `positive_path_utility::Bool=false`: Choice to use a positive path utility translation.
- `negative_path_utility::Bool=false`: Choice to use a negative path utility translation.

# Examples
```julia
generate_diagram!(diagram)
```

!!! note
    The influence diagram must be generated after probabilities and utilities are added
    but before creating the decision model.

!!! note
    If the default probabilities and utilities are not used, define `AbstractPathProbability`
    and `AbstractPathUtility` structures and define P(s), U(s) and U(s, t) functions
    for them. Add the `AbstractPathProbability` and `AbstractPathUtility` structures
    to the influence diagram fields P and U.
"""
function generate_diagram!(diagram::InfluenceDiagram;
    default_probability::Bool=true,
    default_utility::Bool=true,
    positive_path_utility::Bool=false,
    negative_path_utility::Bool=false)

    #Reordering diagram.X to the order of diagram.Names
    diagram.X = OrderedDict(key => diagram.X[key] for key in diagram.Names if haskey(diagram.X, key))

    # Declare P and U if defaults are used
    if default_probability
        C_indices = indices(diagram.C)
        C_I_j_indices = I_j_indices(diagram, diagram.C)
        diagram.P = DefaultPathProbability(C_indices, C_I_j_indices, get_values(diagram.X))
    end

    if default_utility
        V_I_j_indices = I_j_indices(diagram, diagram.V)

        diagram.U = DefaultPathUtility(V_I_j_indices, get_values(diagram.Y))
        if positive_path_utility
            diagram.translation = 1 - sum(minimum.(diagram.U.Y))
        elseif negative_path_utility
            diagram.translation = -1 - sum(maximum.(diagram.U.Y))
        else
            diagram.translation = 0
        end
    end
end


"""
    function indices(dict)

Get the indices of nodes in values of a Dict or OrderedDict.

# Example
```julia-repl
julia> D_indices = indices(diagram.D)
3-element Vector{Int16}:
 3
 6
 9
```
"""
function indices(dict::OrderedDict{K, V}) where {K, V <: AbstractNode}
    indices = Vector{Node}()
    for node in values(dict)
        push!(indices, node.index)
    end
    return indices
end

"""
    function I_j_indices(diagram::InfluenceDiagram, dict)

Get the indices of information sets of nodes in values of a Dict or OrderedDict. Returns Vector{Vector{Node}}.

# Example
```julia-repl
julia> C_I_j_indices = I_j_indices(diagram, diagram.C)
7-element Vector{Vector{Int16}}:
 []
 [1]
 [1, 3]
 [4]
 [4, 6]
 [7]
 [7, 9]
```
"""
function I_j_indices(diagram::InfluenceDiagram, dict::OrderedDict{K, V}) where {K, V <: AbstractNode}
    I_j_indices = Vector{Vector{Node}}()
    for node in values(dict)
        I_j_indices_single_node = Vector{Node}()
        for I_j_node in node.I_j
            push!(I_j_indices_single_node, diagram.Nodes[I_j_node].index)
        end
        push!(I_j_indices, I_j_indices_single_node)
    end
    return I_j_indices
end

"""
    function indices_in_vector(diagram::InfluenceDiagram, nodes::AbstractArray)

Get the indices of an array of nodes and store them in an array.

# Example
```julia-repl
julia> idcs_T1_H2 = indices_of(diagram, ["T1", "H2"])
2-element Vector{Int16}:
 2
 4
```
"""
function indices_in_vector(diagram::InfluenceDiagram, nodes::AbstractArray)
    return [diagram.Nodes[node].index for node in nodes]
end

"""
    function get_values(dict::OrderedDict)

Generic function to get values from an OrderedDict.

# Example
```julia-repl
julia> D_nodes = get_values(diagram.D)
3-element Vector{DecisionNode}:
 DecisionNode("D1", ["T1"], ["treat", "pass"], 3)
 DecisionNode("D2", ["T2"], ["treat", "pass"], 6)
 DecisionNode("D3", ["T3"], ["treat", "pass"], 9)
```
"""
function get_values(dict::OrderedDict)
    return collect(values(dict))
end

"""
    function get_keys(dict::OrderedDict)

Generic function to get keys from an OrderedDict.

# Example
```julia-repl
julia> D_values = get_keys(diagram.D)
3-element Vector{String}:
 "D1"
 "D2"
 "D3"
```
"""
function get_keys(dict::OrderedDict)
    return collect(keys(dict))
end


"""
    function num_states(diagram::InfluenceDiagram, name::Name)

Get the number of states in a given node.

# Example
```julia-repl
julia> NS_O = num_states(diagram, "O")
2
```
"""
function num_states(diagram::InfluenceDiagram, name::Name)
    return get_values(diagram.S)[diagram.Nodes[name].index]
end

# --- ForbiddenPath and FixedPath outer construction functions ---
"""
    function ForbiddenPath(diagram::InfluenceDiagram, nodes::Vector{Name}, paths::Vector{NTuple{N, Name}}) where N

ForbiddenPath outer construction function. Create ForbiddenPath variable.

# Arguments
- `diagram::InfluenceDiagram`: Influence diagram structure
- `nodes::Vector{Name}`: Vector of nodes involved in forbidden paths. Identified by their names.
- `paths`::Vector{NTuple{N, Name}}`: Vector of tuples defining the forbidden combinations of states. States identified by their names.

# Example
```julia
ForbiddenPath(diagram, ["R1", "R2"], [("high", "low"), ("low", "high")])
```
"""
function ForbiddenPath(diagram::InfluenceDiagram, nodes::Vector{Name}, paths::Vector{NTuple{N, Name}}) where N
    node_indices = Vector{Node}()
    for node in nodes
        j = findfirst(i -> i == node, diagram.Names)
        if isnothing(j)
            throw(DomainError("Node $node does not exist."))
        end
        push!(node_indices, j)
    end

    path_set = Set{Path}()
    for s in paths
        s_states = Vector{State}()
        for (i, s_i) in enumerate(s)
            s_i_index = findfirst(x -> x == s_i, get_values(diagram.States)[node_indices[i]])
            if isnothing(s_i_index)
                throw(DomainError("Node $(nodes[i]) does not have a state called $s_i."))
            end

            push!(s_states, s_i_index)
        end
        push!(path_set, Path(s_states))
    end

    return ForbiddenPath((node_indices, path_set))
end

"""
    function FixedPath(diagram::InfluenceDiagram, fixed::Dict{Name, Name})

FixedPath outer construction function. Create FixedPath variable.

# Arguments
- `diagram::InfluenceDiagram`: Influence diagram structure
- `fixed::Dict{Name, Name}`: Dictionary of nodes and their fixed states. Order is node=>state, and both are idefied with their names.

# Example
```julia-repl
julia> fixedpath = FixedPath(diagram, Dict("O" => "lemon"))
Dict{Int16,Int16} with 1 entry:
  1 => 1

julia> vec(collect(paths(states, fixedpath)))
3-element Array{Tuple{Int16,Int16},1}:
 (1, 1)
 (1, 2)
 (1, 3)

```
"""
function FixedPath(diagram::InfluenceDiagram, fixed::Dict{Name, Name})
    fixed_paths = Dict{Node, State}()

    for (j, s_j) in fixed
        j_index = findfirst(i -> i == j, diagram.Names)
        if isnothing(j_index)
            throw(DomainError("Node $j does not exist."))
        end

        s_j_index = findfirst(s -> s == s_j, get_values(diagram.States)[j_index])
        if isnothing(s_j_index)
            throw(DomainError("Node $j does not have a state called $s_j."))
        end
        push!(fixed_paths, Node(j_index) => State(s_j_index))
    end

    return FixedPath(fixed_paths)
end


# --- Local Decision Strategy ---

"""
    LocalDecisionStrategy{N} <: AbstractArray{Int, N}

Local decision strategy type.
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

"""
    function LocalDecisionStrategy(rng::AbstractRNG, diagram::InfluenceDiagram, d::Node)

Generate random decision strategy for decision node `d`.

# Examples
```julia
rng = MersenneTwister(3)
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 5, 2, 3, 2, 2, rand(rng, [2,3], 5))
LocalDecisionStrategy(rng, diagram, diagram.D[1])
```
"""
function LocalDecisionStrategy(rng::AbstractRNG, diagram::InfluenceDiagram, d::Name)
    I_d = diagram.I_j[d]
    states = State[diagram.S[s] for s in I_d]
    state = diagram.S[d]
    data = zeros(Int, states..., state)
    for s in CartesianIndices((states...,))
        s_j = rand(rng, 1:state)
        data[s, s_j] = 1
    end
    LocalDecisionStrategy(diagram.Nodes[d].index, data)
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
