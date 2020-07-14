using Parameters, JuMP
using Base.Iterators: product


# --- Functions ---

"""Iterate over paths."""
function paths(num_states::Vector{T}) where T <: Integer
    product(UnitRange.(one(T), num_states)...)
end

"""Iterate over paths with fixed states."""
function paths(num_states::Vector{T}, fixed::Dict{Int, T}) where T <: Integer
    iters = collect(UnitRange.(one(T), num_states))
    for (i, v) in fixed
        iters[i] = UnitRange(v, v)
    end
    product(iters...)
end

"""Path probability (upper bound)."""
function path_probability(s, C, I_j, X)
    prod(X[j][s[[I_j[j]; j]]...] for j in C)
end

"""Minimum path probability."""
function minimum_path_probability(C, I_j, X, S_j)
    minimum(path_probability(s, C, I_j, X) for s in paths(S_j))
end

"""Total utility of a path."""
function path_utility(s, Y, I_j, V)
    sum(Y[v][s[I_j[v]]...] for v in V)
end


# --- Model ---

"""Defines the DecisionModel type."""
const DecisionModel = Model

"""Influence diagram."""
struct InfluenceDiagram
    C::Vector{Int}
    D::Vector{Int}
    V::Vector{Int}
    A::Vector{Pair{Int, Int}}
    S_j::Vector{Int}
    I_j::Vector{Vector{Int}}
end

"""Construct and validate an influence diagram.

# Arguments
- `C::Vector{Int}`: Change nodes.
- `D::Vector{Int}`: Decision nodes.
- `V::Vector{Int}`: Value nodes.
- `A::Vector{Pair{Int, Int}}`: Arcs between nodes.
- `S_j::Vector{Int}`: Number of states.
"""
function InfluenceDiagram(C::Vector{Int}, D::Vector{Int}, V::Vector{Int}, A::Vector{Pair{Int, Int}}, S_j::Vector{Int})
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
    I_j = [Vector{Int}() for i in 1:N]
    for (i, j) in A
        push!(I_j[j], i)
    end

    InfluenceDiagram(C, D, V, A, S_j, I_j)
end

"""Decision model parameters."""
struct Params
    X::Dict{Int, Array{Float64}}
    Y::Dict{Int, Array{Float64}}
end

"""Construct and validate decision model parameters.

# Arguments
- `diagram::InfluenceDiagram`: The influence diagram associated with the probabilities and consequences.
- `X::Dict{Int, Array{Float64}}`: Probabilities
- `Y::Dict{Int, Array{Float64}}`: Consequences
"""
function Params(diagram::InfluenceDiagram, X::Dict{Int, Array{Float64}}, Y::Dict{Int, Array{Float64}})
    @unpack C, V, S_j, I_j = diagram

    # Validate Probabilities
    for j in C
        S_I = S_j[I_j[j]]
        S_I_j = [S_I; S_j[j]]
        size(X[j]) == Tuple(S_I_j) || error("Array should be dimension |S_I(j)|*|S_j|.")
        all(x > 0 for x in X[j]) || error("Probabilities should be positive.")
        for s_I in paths(S_I)
            sum(X[j][[s_I...; s_j]...] for s_j in 1:S_j[j]) ≈ 1 || error("probabilities shoud sum to one.")
        end
    end

    # Validate consequences
    for j in V
        size(Y[j]) == Tuple(S_j[I_j[j]]) || error("Array should be dimension |S_I(j)|.")
    end

    Params(X, Y)
end

"""Create a multidimensional array of variables."""
function variables(model::Model, dims::Vector{Int}; binary::Bool=false)
    v = Array{VariableRef}(undef, dims...)
    for i in eachindex(v)
        v[i] = @variable(model, binary=binary)
    end
    return v
end

"""Adds a probability sum cut to the model as a lazy constraint."""
function probability_sum_cut(model::DecisionModel, diagram::InfluenceDiagram, params::Params)
    @unpack C, S_j, I_j = diagram
    @unpack X = params
    ϵ = minimum_path_probability(C, I_j, X, S_j)
    # Add the constraints only once
    flag = false
    function probability_sum_cut(cb_data)
        flag && return
        π = model[:π]
        πsum = sum(callback_value(cb_data, π[s]) for s in eachindex(π))
        if !isapprox(πsum, 1.0, atol=ϵ)
            con = @build_constraint(sum(π) == 1.0)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), probability_sum_cut)
end

"""Adds a number of paths cut to the model as a lazy constraint."""
function number_of_paths_cut(model::DecisionModel, diagram::InfluenceDiagram, params::Params, num_paths::Int; atol::Float64 = 0.9)
    @unpack C, S_j, I_j = diagram
    @unpack X = params
    ϵ = minimum_path_probability(C, I_j, X, S_j)
    # Add the constraints only once
    flag = false
    function number_of_paths_cut(cb_data)
        flag && return
        π = model[:π]
        πnum = sum(callback_value(cb_data, π[s]) ≥ ϵ for s in eachindex(π))
        if !isapprox(πnum, num_paths, atol = atol)
            con = @build_constraint(sum(π[s...] / path_probability(s, C, I_j, X) for s in paths(S_j)) == num_paths)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), number_of_paths_cut)
end

"""Construct a DecisionModel from an influence diagram and parameters.

# Arguments
- `diagram::InfluenceDiagram`
- `params::Params`
"""
function DecisionModel(diagram::InfluenceDiagram, params::Params)
    @unpack C, D, V, A, S_j, I_j = diagram
    @unpack X, Y = params

    # Affine transformation to positive utility function: Normalize plus one.
    v_min = minimum(minimum(v) for (k, v) in Y)
    v_max = maximum(maximum(v) for (k, v) in Y)
    Y′ = Dict(k => (@. (v - v_min)/(v_max - v_min) + 1) for (k, v) in Y)

    # Initialize the model
    model = DecisionModel()

    # --- Variables ---
    π = variables(model, S_j)
    z = Dict{Int, Array{VariableRef}}(
        j => variables(model, S_j[[I_j[j]; j]]; binary=true) for j in D)

    # Add variable to the model.obj_dict.
    model[:π] = π
    model[:z] = z

    # --- Objectives ---
    @objective(model, Max, sum(π[s...] * path_utility(s, Y′, I_j, V) for s in paths(S_j)))

    # --- Constraints ---
    for j in D, s_I in paths(S_j[I_j[j]])
        @constraint(model, sum(z[j][[s_I...; s_j]...] for s_j in 1:S_j[j]) == 1)
    end

    for s in paths(S_j)
        @constraint(model, 0 ≤ π[s...] ≤ path_probability(s, C, I_j, X))
    end

    for s in paths(S_j), j in D
        @constraint(model, π[s...] ≤ z[j][s[[I_j[j]; j]]...])
    end

    return model
end
