using Parameters, JuMP
using Base.Iterators: product


# --- Functions ---

"""Generate all paths."""
function paths(lengths::Vector{T}) where T <: Integer
    product(UnitRange{T}.(1, lengths)...)
end


# --- Model ---

"""Defines the DecisionModel type."""
const DecisionModel = Model

"""Specification for different model scenarios. For example, we can specify toggling on and off constraints and objectives.

# Arguments
- `probability_sum_cut::Bool`: Toggle probability sum cuts on and off.
- `num_paths::Int`: If larger than zero, enables the number of paths cuts using the supplied value.
"""
@with_kw struct Specs
    probability_sum_cut::Bool = false
    num_paths::Int = 0
end

"""Decision graph is a directed, acyclic graph.

# Arguments
- `C::Vector{Int}`: Change nodes.
- `D::Vector{Int}`: Decision nodes.
- `V::Vector{Int}`: Value nodes.
- `A::Vector{Pair{Int, Int}}`: Arcs between nodes.
- `S_j::Vector{Int}`: Number of states.
- `I_j::Vector{Vector{Int}}`: Information set.
"""
struct DecisionGraph
    C::Vector{Int}
    D::Vector{Int}
    V::Vector{Int}
    A::Vector{Pair{Int, Int}}
    S_j::Vector{Int}
    I_j::Vector{Vector{Int}}
end

"""Construct and validate a decision graph."""
function DecisionGraph(C::Vector{Int}, D::Vector{Int}, V::Vector{Int}, A::Vector{Pair{Int, Int}}, S_j::Vector{Int})
    # Enforce sorted and unique elements.
    C = sort(unique(C))
    D = sort(unique(D))
    V = sort(unique(V))
    A = sort(unique(A))

    # Sizes
    n = length(C) + length(D)
    n_V = length(V)

    ## Validate nodes
    isempty(C ∩ D) || error("The sets of change and decision nodes should be disjoint.")
    Set(C ∪ D) == Set(1:n) || error("Union of change and decision nodes should be {1,...,n}.")
    Set(V) == Set((n+1):(n+n_V)) || error("Values nodes should be {n+1,...,n+n_V}.")

    ## Validate arcs
    # 1) Inclusion A ⊆ N×N.
    # 2) Graph is acyclic.
    # 3) There are no arcs from chance or decision nodes to value nodes.
    all(1 ≤ i < j ≤ (n+n_V) for (i, j) in A) || error("Forall (i,j)∈A we should have 1≤i<j≤n+n_V.")

    ## Validate states
    # Each chance and decision node has a finite number of states
    length(S_j) == n || error("Each change and decision node should have states.")
    all(S_j[j] ≥ 1 for j in 1:n) || error("Each change and decision node should have finite number of states.")

    # Construct the information set
    I_j = [Vector{Int}() for i in 1:(n+n_V)]
    for (i, j) in A
        push!(I_j[j], i)
    end

    DecisionGraph(C, D, V, A, S_j, I_j)
end

"""Model parameters.

# Arguments
- `X`: Probabilities, X[j][s_I(j);s_j], ∀j∈C
- `Y`: Consequences, Y[j][s_I(j)], ∀j∈V
- `U`: Utilities map consequences to real valued outcomes.
"""
struct Params
    X::Dict{Int, Array{Float64}}
    Y::Dict{Int, Array{Int}}
    U::Vector{Float64}
end

"""Construct and validate model parameters."""
function Params(graph::DecisionGraph, X::Dict{Int, Array{Float64}}, Y::Dict{Int, Array{Int}}, U::Vector{Float64})
    @unpack C, V, S_j, I_j = graph

    # Validate Probabilities
    for j in C
        S_I = S_j[I_j[j]]
        S_I_j = [S_I; S_j[j]]
        size(X[j]) == Tuple(S_I_j) || error("Array should be dimension |S_I(j)|*|S_j|.")
        all(x ≥ 0 for x in X[j]) || error("Probabilities should be positive.")
        for s_I in paths(S_I)
            sum(X[j][[s_I...; s]...] for s in 1:S_j[j]) ≈ 1 || error("probabilities shoud sum to one.")
        end
    end

    # Validate consequences
    for j in V
        size(Y[j]) == Tuple(S_j[I_j[j]]) || error("Array should be dimension |S_I(j)|.")
    end

    # Validate utilities
    length(U) == sum(prod(S_j[j] for j in I_j[i]) for i in V) || error("")

    Params(X, Y, U)
end

"""Probability sum lazy cut."""
function probability_sum_cut(cb_data, model, π, ϵ)
    # TODO: add only once
    πsum = sum(callback_value(cb_data, π[s]) for s in eachindex(π))
    if !isapprox(πsum, 1.0, atol=ϵ)
        con = @build_constraint(sum(π) == 1.0)
        MOI.submit(model, MOI.LazyConstraint(cb_data), con)
    end
end

"""Number of paths lazy cut."""
function number_of_paths_cut(cb_data, model, π, ϵ, p, num_paths, S_j)
    # TODO: add only once
    πnum = sum(callback_value(cb_data, π[s]) >= ϵ for s in eachindex(π))
    if !isapprox(πnum, num_paths, atol = 0.9)
        con = @build_constraint(sum(π[s] / p(s) for s in paths(S_j)) == num_paths)
        MOI.submit(model, MOI.LazyConstraint(cb_data), con)
    end
end

"""Initializes the DecisionModel."""
function DecisionModel(specs::Specs, graph::DecisionGraph, params::Params)
    @unpack C, D, V, A, S_j, I_j = graph
    @unpack X, Y, U = params

    """Upper bound of probability of a path."""
    probability(s) = prod(X[j][s[[I_j[j]; j]]...] for j in C)

    # Minimum path probability
    ϵ = minimum(probability(s) for s in paths(S_j))

    # Affine transformion to non-negative utility function.
    U′ = U .- minimum(U)

    """Total utility of a path."""
    utility(s) = sum(U′[Y[v][s[I_j[v]]...]] for v in V)

    # Initialize the model
    model = DecisionModel()

    # --- Variables ---
    π = fill(VariableRef(model), S_j...)
    for s in paths(S_j)
        π[s...] = @variable(model, base_name="π[$(s)]")
    end

    z = Dict{Int, Array{VariableRef}}()
    for j in D
        S_I_j = S_j[[I_j[j]; j]]
        z[j] = fill(VariableRef(model), S_I_j...)
        for s in paths(S_I_j)
            z[j][s...] = @variable(model, binary=true, base_name="z[$j,$(s)]")
        end
    end

    # --- Objectives ---
    @expression(model, expected_utility,
        sum(π[s...] * utility(s) for s in paths(S_j)))
    @objective(model, Max, expected_utility)

    # --- Constraints ---
    for j in D
        for s_I in paths(S_j[I_j[j]])
            @constraint(model, sum(z[j][[s_I...; s]...] for s in 1:S_j[j]) == 1)
        end
    end

    for s in paths(S_j)
        @constraint(model, 0 ≤ π[s...] ≤ probability(s))
        for j in D
            @constraint(model, π[s...] ≤ z[j][s[[I_j[j]; j]]...])
        end
    end

    if specs.probability_sum_cut
        MOI.set(
            model, MOI.LazyConstraintCallback(),
            cb_data -> probability_sum_cut(cb_data, model, π, ϵ))
    end

    if specs.num_paths > 0
        MOI.set(
            model, MOI.LazyConstraintCallback(),
            cb_data -> number_of_paths_cut(cb_data, model, π, ϵ, probability, specs.num_paths, S_j))
    end

    return model
end
