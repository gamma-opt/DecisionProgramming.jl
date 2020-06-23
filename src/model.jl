using Parameters, JuMP
using DataStructures: SortedSet
using Base.Iterators: product


# --- Model ---

"""Defines the DecisionModel type."""
const DecisionModel = Model

"""Specification for different model scenarios. For example, we can specify toggling on and off certain constraints and objectives.
"""
@with_kw struct Specs
    lazy_constraints::Bool
end

"""Directed, acyclic graph."""
struct DecisionGraph
    C::Vector{Int} # Change nodes
    D::Vector{Int} # Decision nodes
    V::Vector{Int} # Value nodes
    A::Vector{Pair{Int, Int}} # Arcs
    S_j::Vector{Int} # Number of states per node j∈C∪D
    I_j::Vector{Vector{Int}} # Information set
end

"""Validate decision graph."""
function DecisionGraph(C::SortedSet{Int}, D::SortedSet{Int}, V::SortedSet{Int}, A::Vector{Pair{Int, Int}}, S_j::Vector{Int})
    # Sizes
    n = length(C) + length(D)
    n_V = length(V)

    ## Validate nodes
    isempty(C ∩ D) || error("Change and decision nodes are not disjoint.")
    C ∪ D == SortedSet(1:n) || error("Union of change and decision nodes should be 1:n.")
    V == SortedSet((n+1):(n+n_V)) || error("Values nodes should be (n+1):(n+n_V).")

    ## Validate arcs
    # 1) Inclusion A ⊆ N×N.
    # 2) Graph is acyclic.
    # 3) There are no arcs from chance or decision nodes to value nodes.
    all(1 ≤ i < j ≤ (n+n_V) for (i, j) in A) || error("")

    ## Validate states
    # Each chance and decision node has a finite number of states
    length(S_j) == n || error("")
    all(S_j[j] ≥ 1 for j in 1:n) || error("")

    # Construction the information set
    I_j = [Vector{Int}() for i in 1:(n+n_V)]
    for (i, j) in A
        push!(I_j[j], i)
    end
    I_j = sort.(I_j)

    DecisionGraph(collect(C), collect(D), collect(V), A, S_j, I_j)
end

"""Sorted sets enforce unique and sorted indices."""
function DecisionGraph(C::Vector{Int}, D::Vector{Int}, V::Vector{Int}, A::Vector{Pair{Int, Int}}, S_j::Vector{Int})
    DecisionGraph(SortedSet(C), SortedSet(D), SortedSet(V), A, S_j)
end

# j => X[s_I(j);s_j], ∀j∈C
# each array is dimension S_I(j)×S_j
"""Probabilities"""
struct Probabilities
    X::Dict{Int, Array{Float64}}
end

# j => Y[s_I(j)], ∀j∈V
# each array is dimension S_I(j)
"""Utilities"""
struct Utilities
    Y::Dict{Int, Array{Float64}}
end

"""Validate probabilities"""
function Probabilities(graph::DecisionGraph, X::Dict{Int, Array{Float64}})
    @unpack C, S_j, I_j = graph
    # All nodes j∈C are assigned a probability based on its information set
    for j in C
        # Test dimensions
        S_I = [S_j[i] for i in I_j[j]]
        S_I_j = [S_I; S_j[j]]
        size(X[j]) == Tuple(S_I_j) || error("")
        # # All probabilities are positive
        all(x ≥ 0 for x in X[j]) || error("")
        # Probabilities sum to one
        for s_I in product(UnitRange.(1, S_I)...)
            sum(X[j][[s_I...; s]...] for s in 1:S_j[j]) ≈ 1 || error("")
        end
    end
    Probabilities(X)
end

"""Validate utilities"""
function Utilities(graph::DecisionGraph, Y::Dict{Int, Array{Float64}})
    @unpack V, S_j, I_j = graph
    # All nodes v∈V are assigned a utility based on its information set
    for j in V
        S_I = [S_j[i] for i in I_j[j]]
        size(Y[j]) == Tuple(S_I) || error("")
    end
    Utilities(Y)
end

"""Initializes the DecisionModel."""
function DecisionModel(specs::Specs, graph::DecisionGraph, probabilities::Probabilities, utilities::Utilities)
    @unpack C, D, V, A, S_j, I_j = graph
    @unpack X = probabilities
    @unpack Y = utilities

    # Initialize the model
    model = DecisionModel()

    # Variables
    π = fill(VariableRef(model), S_j...)
    for s in CartesianIndices(π)
        π[s] = @variable(model, base_name="π[$(Tuple(s))]")
    end

    z = Dict{Int, Array{VariableRef}}()
    for j in D
        S_I = [S_j[i] for i in I_j[j]]
        S_I_j = [S_I; S_j[j]]
        z[j] = fill(VariableRef(model), S_I_j...)
        for s in CartesianIndices(z[j])
            z[j][s] = @variable(model, binary=true, base_name="z[$j,$(Tuple(s))]")
        end
    end

    # Objectives
    expected_utility = AffExpr(0)
    for s in CartesianIndices(π)
        for v in V
            s_I = [s[i] for i in I_j[v]]
            U_s = Y[v][s_I...]
            add_to_expression!(expected_utility, π[s] * U_s)
        end
    end
    @objective(model, Max, expected_utility)

    # Constraints
    for j in D
        S_I = [S_j[i] for i in I_j[j]]
        for s_I in product(UnitRange.(1, S_I)...)
            @constraint(model, sum(z[j][[s_I...; s]...] for s in 1:S_j[j]) == 1)
        end
    end

    for s in CartesianIndices(π)
        p_s = 1
        for j in C
            S_I_j = [s[i] for i in [I_j[j]...; j]]
            p_s *= X[j][S_I_j...]
        end
        @constraint(model, 0≤π[s]≤p_s)
    end

    for s in CartesianIndices(π)
        for j in D
            S_I_j = [s[i] for i in [I_j[j]...; j]]
            @constraint(model, π[s]≤z[j][S_I_j...])
        end
    end

    if specs.lazy_constraints
        # TODO:
    end

    return model
end
