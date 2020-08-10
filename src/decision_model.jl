using Parameters, JuMP

"""Positive affine transformation of path utility. Normalized to into range from 1 to 2.

# Examples
```julia
U⁺ = PositivePathUtility(S, U)
```
"""
struct PositivePathUtility <: AbstractPathUtility
    U::AbstractPathUtility
    min::Float64
    max::Float64
    function PositivePathUtility(S::States, U::AbstractPathUtility)
        (u_min, u_max) = extrema(U(s) for s in paths(S))
        new(U, u_min, u_max)
    end
end

"""Evaluate positive affine transformation of the path utility.

# Examples
```julia-repl
julia> 1 ≤ U⁺(s) ≤ 2
true
```
"""
(U::PositivePathUtility)(s::Path) = (U.U(s) - U.min)/(U.max - U.min) + 1

"""Create a multidimensional array of JuMP variables.

# Examples
```julia
model = Model()
v1 = variables(model, [2, 3, 2])
v2 = variables(model, [2, 3, 2]; binary=true)
```
"""
function variables(model::Model, dims::Vector{Int}; binary::Bool=false)
    v = Array{VariableRef}(undef, dims...)
    for i in eachindex(v)
        v[i] = @variable(model, binary=binary)
    end
    return v
end

"""DecisionModel type. Alias for `JuMP.Model`."""
const DecisionModel = Model

"""Construct a DecisionModel from states, decision nodes and path probability.

# Examples
```julia
model = DecisionModel(S, D, P; positive_path_utility=true)
```
"""
function DecisionModel(S::States, D::Vector{DecisionNode}, P::AbstractPathProbability; positive_path_utility::Bool=true)
    model = DecisionModel()

    π = variables(model, S[:])
    z = [variables(model, S[[d.I_j; d.j]]; binary=true) for d in D]

    for (d, z_j) in zip(D, z)
        for s_I in paths(S[d.I_j])
            @constraint(model, sum(z_j[s_I..., s_j] for s_j in 1:S[d.j]) == 1)
        end
    end

    for s in paths(S)
        @constraint(model, 0 ≤ π[s...] ≤ P(s))
    end

    for s in paths(S)
        for (d, z_j) in zip(D, z)
            @constraint(model, π[s...] ≤ z_j[s[[d.I_j; d.j]]...])
        end
    end

    if !positive_path_utility
        for s in paths(S)
            @constraint(model,
                π[s...] ≥ P(s) + sum(z_j[s[[d.I_j; d.j]]...] for (d, z_j) in zip(D, z)) - length(D))
        end
    end

    model[:π] = π
    model[:z] = z

    return model
end

"""Adds a probability sum cut to the model as a lazy constraint.

# Examples
```julia
probability_sum_cut(model, S, P)
```
"""
function probability_sum_cut(model::DecisionModel, S::States, P::AbstractPathProbability)
    # Add the constraints only once
    ϵ = minimum(P(s) for s in paths(S))
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

"""Adds a number of paths cut to the model as a lazy constraint.

# Examples
```julia
atol = 0.9  # Tolerance to trigger the creation of the lazy cut
number_of_paths_cut(model, S, P; atol=atol)
```
"""
function number_of_paths_cut(model::DecisionModel, S::States, P::AbstractPathProbability; atol::Float64 = 0.9)
    ϵ = minimum(P(s) for s in paths(S))
    num_active_paths = prod(S[c.j] for c in P.C)
    # Add the constraints only once
    flag = false
    function number_of_paths_cut(cb_data)
        flag && return
        π = model[:π]
        πnum = sum(callback_value(cb_data, π[s]) ≥ ϵ for s in eachindex(π))
        if !isapprox(πnum, num_active_paths, atol = atol)
            con = @build_constraint(sum(π[s...] / P(s) for s in paths(S)) == num_active_paths)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), number_of_paths_cut)
end


# --- Objective Functions ---

"""Create an expected value objective.

# Examples
```julia
EV = expected_value(model, S, U)
```
"""
function expected_value(model::DecisionModel, S::States, U::AbstractPathUtility)
    @expression(model, sum(model[:π][s...] * U(s) for s in paths(S)))
end

"""Create a conditional value-at-risk (CVaR) objective.

# Examples
```julia
α = 0.05  # Parameter such that 0 ≤ α ≤ 1
CVaR = conditional_value_at_risk(model, S, U, α)
```
"""
function conditional_value_at_risk(model::DecisionModel, S::States, U::AbstractPathUtility, α::Float64)
    0 ≤ α ≤ 1 || error("α should be 0 ≤ α ≤ 1")

    # Pre-computer parameters
    u = collect(Iterators.flatten(U(s) for s in paths(S)))
    u_sorted = sort(u)
    u_min = u_sorted[1]
    u_max = u_sorted[end]
    M = u_max - u_min
    ϵ = minimum(filter(!iszero, abs.(diff(u_sorted)))) / 2

    # Variables
    η = @variable(model)
    λ = variables(model, S[:]; binary=true)
    λ_bar = variables(model, S[:]; binary=true)
    ρ = variables(model, S[:])
    ρ_bar = variables(model, S[:])

    # Constraints
    π = model[:π]
    @constraint(model, u_min ≤ η ≤ u_max)
    for s in paths(S)
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
    @constraint(model, sum(ρ_bar[s...] for s in paths(S)) == α)

    # Add variables to the model
    model[:η] = η
    model[:ρ] = ρ
    model[:ρ_bar] = ρ_bar

    # Return CVaR as an expression
    CVaR = @expression(model, sum(ρ_bar[s...] * U(s) for s in paths(S)) / α)

    return CVaR
end


# --- Decision Strategy ---

"""Decision strategy type."""
struct LocalDecisionStrategy{N} <: AbstractArray{Int, N}
    data::Array{Int, N}
    function LocalDecisionStrategy(data::Array{Int, N}) where N
        all(0 ≤ x ≤ 1 for x in data) || error("All values x must be 0 ≤ x ≤ 1.")
        for s_I in CartesianIndices(size(data)[1:end-1])
            sum(data[s_I, :]) == 1 || error("")
        end
        new{N}(data)
    end
end

Base.size(Z::LocalDecisionStrategy) = size(Z.data)
Base.IndexStyle(::Type{<:LocalDecisionStrategy}) = IndexLinear()
Base.getindex(Z::LocalDecisionStrategy, i::Int) = getindex(Z.data, i)
Base.getindex(Z::LocalDecisionStrategy, I::Vararg{Int,N}) where N = getindex(Z.data, I...)

"""Construct decision strategy from variable refs."""
function LocalDecisionStrategy(z::Array{VariableRef})
    LocalDecisionStrategy(@. Int(round(value(z))))
end

"""Evalute decision strategy."""
(Z::LocalDecisionStrategy)(s_I::Path)::State = findmax(Z[s_I..., :])[2]


"""Global decision strategy type."""
struct DecisionStrategy
    D::Vector{DecisionNode}
    Z_j::Vector{LocalDecisionStrategy}
end

"""Extract values for decision variables from solved decision model.

# Examples
```julia
Z = GlobalDecisionStrategy(model, D)
```
"""
function DecisionStrategy(model::DecisionModel, D::Vector{DecisionNode})
    DecisionStrategy(D, [LocalDecisionStrategy(v) for v in model[:z]])
end
