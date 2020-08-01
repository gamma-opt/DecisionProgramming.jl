using Parameters, JuMP

"""Create a multidimensional array of JuMP variables."""
function variables(model::Model, dims::Vector{Int}; binary::Bool=false)
    v = Array{VariableRef}(undef, dims...)
    for i in eachindex(v)
        v[i] = @variable(model, binary=binary)
    end
    return v
end

"""DecisionModel type."""
const DecisionModel = Model

"""Construct a DecisionModel from an influence diagram and probabilities.

# Examples
```julia
model = DecisionModel(G, P; positive_path_utility=true)
```
"""
function DecisionModel(G::InfluenceDiagram, P::PathProbability; positive_path_utility::Bool=true)
    @unpack D, S_j, I_j = G

    model = DecisionModel()

    π = variables(model, S_j)
    z = Dict{Int, Array{VariableRef}}(
        j => variables(model, S_j[[I_j[j]; j]]; binary=true) for j in D)

    for j in D, s_I in paths(S_j[I_j[j]])
        @constraint(model, sum(z[j][[s_I...; s_j]...] for s_j in 1:S_j[j]) == 1)
    end

    for s in paths(S_j)
        @constraint(model, 0 ≤ π[s...] ≤ P(s))
    end

    for s in paths(S_j), j in D
        @constraint(model, π[s...] ≤ z[j][s[[I_j[j]; j]]...])
    end

    if !positive_path_utility
        for s in paths(S_j)
            @constraint(model,
                π[s...] ≥ P(s) + sum(z[j][s[[I_j[j]; j]]...] for j in D) - length(D))
        end
    end

    model[:π] = π
    model[:z] = z

    return model
end

"""Adds a probability sum cut to the model as a lazy constraint.

# Examples
```julia
probability_sum_cut(model, P)
```
"""
function probability_sum_cut(model::DecisionModel, P::PathProbability)
    # Add the constraints only once
    flag = false
    function probability_sum_cut(cb_data)
        flag && return
        π = model[:π]
        πsum = sum(callback_value(cb_data, π[s]) for s in eachindex(π))
        if !isapprox(πsum, 1.0, atol=P.min)
            con = @build_constraint(sum(π) == 1.0)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), probability_sum_cut)
end

"""Adds a number of paths cut to the model as a lazy constraint.

# Examples
```julia
atol = 0.9  # Tolerance to trigger the creation of the lazy cut
number_of_paths_cut(model, G, P; atol=atol)
```
"""
function number_of_paths_cut(model::DecisionModel, G::InfluenceDiagram, P::PathProbability; atol::Float64 = 0.9)
    num_active_paths = prod(G.S_j[G.C])
    # Add the constraints only once
    flag = false
    function number_of_paths_cut(cb_data)
        flag && return
        π = model[:π]
        πnum = sum(callback_value(cb_data, π[s]) ≥ P.min for s in eachindex(π))
        if !isapprox(πnum, num_active_paths, atol = atol)
            con = @build_constraint(sum(π[s...] / P(s) for s in paths(G.S_j)) == num_active_paths)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), number_of_paths_cut)
end


# --- Objective Functions ---

"""Expected value objective.

# Examples
```julia
EV = expected_value(model, G, U)
```
"""
function expected_value(model::DecisionModel, G::InfluenceDiagram, U::PathUtility)
    @expression(model, sum(model[:π][s...] * positive_affine(U, s) for s in paths(G.S_j)))
end

"""Conditional value-at-risk (CVaR) objective. Also known as Expected Shortfall (ES).

# Examples
```julia
α = 0.05  # Parameter such that 0 ≤ α ≤ 1
ES = conditional_value_at_risk(model, G, U, α)
```
"""
function conditional_value_at_risk(model::DecisionModel, G::InfluenceDiagram, U::PathUtility, α::Float64)
    @unpack S_j = G
    0 ≤ α ≤ 1 || error("α should be 0 ≤ α ≤ 1")

    # Pre-computer parameters
    u = collect(Iterators.flatten(positive_affine(U, s) for s in paths(S_j)))
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
        u_s = positive_affine(U, s)
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
    return @expression(model, sum(ρ_bar[s...] * positive_affine(U, s) for s in paths(S_j)) / α)
end


# --- Decision Strategy ---

"""Decision strategy type."""
struct DecisionStrategy
    Z::Dict{Node, Array{State, N} where N}
end

(d::DecisionStrategy)(j::Node) = d.Z[j]
(d::DecisionStrategy)(j::Node, s::Path) = findmax(d.Z[j][s..., :])[2]
(d::DecisionStrategy)(j::Node, s::Path, G::InfluenceDiagram) = d(j, s[G.I_j[j]])

"""Extract values for decision variables from a decision model.

# Examples
```julia
Z = DecisionStrategy(model)
```
"""
function DecisionStrategy(model::DecisionModel)
    DecisionStrategy(Dict(i => (@. Int(round(value(v)))) for (i, v) in model[:z]))
end
