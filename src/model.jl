using Parameters, JuMP
using Base.Iterators: product


# --- Types ---

"""Influence diagram."""
struct InfluenceDiagram
    C::Vector{Int}
    D::Vector{Int}
    V::Vector{Int}
    A::Vector{Pair{Int, Int}}
    S_j::Vector{Int}
    I_j::Vector{Vector{Int}}
end

"""Probabilities type."""
const Probabilities = Dict{Int, Array{Float64}}

"""Consequences type."""
const Consequences = Dict{Int, Array{Float64}}

"""UtilityFunction type. Maps path to real."""
const UtilityFunction = Function

"""Defines the DecisionModel type."""
const DecisionModel = Model

"""Decision strategy type."""
const DecisionStrategy = Dict{Int, Array{Int}}


# --- Functions ---

"""Iterate over paths."""
function paths(num_states::Vector{Int})
    product(UnitRange.(one(Int), num_states)...)
end

"""Iterate over paths with fixed states."""
function paths(num_states::Vector{Int}, fixed::Dict{Int, Int})
    iters = collect(UnitRange.(one(Int), num_states))
    for (i, v) in fixed
        iters[i] = UnitRange(v, v)
    end
    product(iters...)
end

"""Path probability (upper bound)."""
function path_probability(s::NTuple{N, Int}, G::InfluenceDiagram, X::Probabilities) where N
    @unpack C, I_j = G
    prod(X[j][s[[I_j[j]; j]]...] for j in C)
end


# --- Model ---

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

"""Validate probabilities."""
function validate_probabilities(G::InfluenceDiagram, X::Probabilities)::Probabilities
    @unpack C, S_j, I_j = G
    for j in C
        S_I = S_j[I_j[j]]
        S_I_j = [S_I; S_j[j]]
        size(X[j]) == Tuple(S_I_j) || error("Array should be dimension |S_I(j)|*|S_j|.")
        all(x > 0 for x in X[j]) || error("Probabilities should be positive.")
        for s_I in paths(S_I)
            sum(X[j][[s_I...; s_j]...] for s_j in 1:S_j[j]) ≈ 1 || error("probabilities shoud sum to one.")
        end
    end
    return X
end

"""Validate consequences."""
function validate_consequences(G::InfluenceDiagram, Y::Consequences)::Consequences
    @unpack V, S_j, I_j = G
    for j in V
        size(Y[j]) == Tuple(S_j[I_j[j]]) || error("Array should be dimension |S_I(j)|.")
    end
    return Y
end

"""Create a multidimensional array of variables."""
function variables(model::Model, dims::Vector{Int}; binary::Bool=false)
    v = Array{VariableRef}(undef, dims...)
    for i in eachindex(v)
        v[i] = @variable(model, binary=binary)
    end
    return v
end

"""Construct a DecisionModel from an influence diagram and parameters.

# Arguments
- `G::InfluenceDiagram`
- `X::Probabilities`
- `positive_path_utility::Bool=true`
"""
function DecisionModel(G::InfluenceDiagram, X::Probabilities; positive_path_utility::Bool=true)
    @unpack C, D, S_j, I_j = G

    model = DecisionModel()

    π = variables(model, S_j)
    z = Dict{Int, Array{VariableRef}}(
        j => variables(model, S_j[[I_j[j]; j]]; binary=true) for j in D)

    for j in D, s_I in paths(S_j[I_j[j]])
        @constraint(model, sum(z[j][[s_I...; s_j]...] for s_j in 1:S_j[j]) == 1)
    end

    for s in paths(S_j)
        @constraint(model, 0 ≤ π[s...] ≤ path_probability(s, G, X))
    end

    for s in paths(S_j), j in D
        @constraint(model, π[s...] ≤ z[j][s[[I_j[j]; j]]...])
    end

    if !positive_path_utility
        for s in paths(S_j)
            @constraint(model,
                π[s...] ≥ path_probability(s, G, X) +
                          sum(z[j][s[[I_j[j]; j]]...] for j in D) - length(D))
        end
    end

    model[:π] = π
    model[:z] = z

    return model
end

"""Extract values for decision variables from a decision model."""
function DecisionStrategy(model::DecisionModel)
    DecisionStrategy(i => (@. Int(round(value(v)))) for (i, v) in model[:z])
end

"""Adds a probability sum cut to the model as a lazy constraint."""
function probability_sum_cut(model::DecisionModel, G::InfluenceDiagram, X::Probabilities)
    @unpack C, S_j, I_j = G
    ϵ = minimum(path_probability(s, G, X) for s in paths(S_j))
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
function number_of_paths_cut(model::DecisionModel, G::InfluenceDiagram, X::Probabilities, num_paths::Int; atol::Float64 = 0.9)
    @unpack C, S_j, I_j = G
    ϵ = minimum(path_probability(s, G, X) for s in paths(S_j))
    # Add the constraints only once
    flag = false
    function number_of_paths_cut(cb_data)
        flag && return
        π = model[:π]
        πnum = sum(callback_value(cb_data, π[s]) ≥ ϵ for s in eachindex(π))
        if !isapprox(πnum, num_paths, atol = atol)
            con = @build_constraint(sum(π[s...] / path_probability(s, G, X) for s in paths(S_j)) == num_paths)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), number_of_paths_cut)
end

"""Affine positive tranformation of path utility."""
function transform_affine_positive(U::UtilityFunction, S_j::Array{Int})
    (u_min, u_max) = extrema(U(s) for s in paths(S_j))
    return s -> (U(s) - u_min)/(u_max - u_min) + 1
end

"""Expected value."""
function expected_value(model::DecisionModel, U::UtilityFunction, S_j::Vector{Int})
    @expression(model, sum(model[:π][s...] * U(s) for s in paths(S_j)))
end

"""Value-at-risk."""
function value_at_risk(model::DecisionModel, U::UtilityFunction, S_j::Vector{Int}, α::Float64)
    # Pre-computer parameters
    u = collect(Iterators.flatten(U(s) for s in paths(S_j)))
    u_sorted = sort(u)
    u_min = u_sorted[1]
    u_max = u_sorted[end]
    M = u_max - u_min
    ϵ = minimum(filter(!iszero, abs.(diff(u_sorted)))) / 2

    # Variables
    η = @variable(model)
    λ = variables(model, S_j; binary=true)
    λ_bar = variables(model, S_j; binary=true)
    ρ = variables(model, S_j)
    ρ_bar = variables(model, S_j)

    # Constraints
    π = model[:π]
    @constraint(model, u_min ≤ η ≤ u_max)
    for s in paths(S_j)
        u_s = U(s)
        @constraint(model, η - u_s ≤ M * λ[s...])
        @constraint(model, η - u_s ≥ (M + ϵ) * λ[s...] - M)
        @constraint(model, η - u_s ≤ (M + ϵ) * λ_bar[s...] - ϵ)
        @constraint(model, η - u_s ≥ M * (λ_bar[s...] - 1))
        @constraint(model, 0 ≤ ρ[s...])
        @constraint(model, 0 ≤ ρ_bar[s...])
        @constraint(model, ρ[s...] ≤ λ[s...])
        @constraint(model, ρ_bar[s...] ≤ λ_bar[s...])
        @constraint(model, ρ[s...] ≤ ρ_bar[s...])
        @constraint(model, ρ_bar[s...] ≤ π[s...])
        @constraint(model, π[s...] - (1 - λ[s...]) ≤ ρ[s...])
    end
    @constraint(model, sum(ρ_bar[s...] for s in paths(S_j)) == α)

    # Add variables to the model
    model[:η] = η
    model[:ρ] = ρ
    model[:ρ_bar] = ρ_bar

    # Return CVaR as an expression
    return @expression(model, sum(ρ_bar[s...] * U(s) for s in paths(S_j)) / α)
end
