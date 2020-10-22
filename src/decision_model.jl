using JuMP

"""DecisionVariables type."""
const DecisionVariables = Vector{<:Array{VariableRef}}

"""Create decision variables and constraints.

# Examples
```julia
z = decision_variables(model, S, D)
```
"""
function decision_variables(model::Model, S::States, D::Vector{DecisionNode}; names::Bool=false, name::String="z")
    z = Vector{Array{VariableRef}}()
    for d in D
        # Create decision variables.
        dims = S[[d.I_j; d.j]]
        z_j = Array{VariableRef}(undef, dims...)
        for s in paths(dims)
            z_j[s...] = @variable(model, binary=true, base_name=(names ? "$(name)_$(d.j)$(s)" : ""))
        end
        # Constraints to one decision per decision strategy.
        for s_I in paths(S[d.I_j])
            @constraint(model, sum(z_j[s_I..., s_j] for s_j in 1:S[d.j]) == 1)
        end
        push!(z, z_j)
    end
    return z
end

"""PathProbabilityVariables type."""
const PathProbabilityVariables{N} = Dict{NTuple{N, Int}, VariableRef} where N

"""Create path probability variables and constraints.

# Examples
```julia
π_s = path_probability_variables(model, z, S, D, P)
π_s = path_probability_variables(model, z, S, D, P; hard_lower_bound=false))
```
"""
function path_probability_variables(model::Model, z::DecisionVariables, S::States, D::Vector{DecisionNode}, P::AbstractPathProbability; hard_lower_bound::Bool=true, names::Bool=false, name::String="π_s", nocomb::Dict{}=Dict())
    N = length(S)
    π_s = Dict{NTuple{N, Int}, VariableRef}()

    # Iterate over all paths. Skip paths with path probability of zero.
    for s in paths(S)
        keep_s = true
        for k in keys(nocomb)
            if collect(s[k]) in nocomb[k]
                keep_s = false
            end
        end
        if iszero(P(s)) || !keep_s
            continue
        end
        
        # Create a path probability variable
        π = @variable(model, base_name=(names ? "$(name)$(s)" : ""))
        π_s[s] = π

        # Soft constraint on the lower bound.
        @constraint(model, π ≥ 0)

        # Hard constraint on the upper bound.
        @constraint(model, π ≤ P(s))

        # Constraints the path probability to zero if the path is
        # incompatible with the decision strategy.
        for (d, z_j) in zip(D, z)
            @constraint(model, π ≤ z_j[s[[d.I_j; d.j]]...])
        end

        # Hard constraint on the lower bound.
        if hard_lower_bound
            n_z = @expression(model, sum(z_j[s[[d.I_j; d.j]]...] for (d, z_j) in zip(D, z)))
            @constraint(model, π ≥ P(s) + n_z - length(D))
        end
    end
    return π_s
end

"""Adds a probability cut to the model as a lazy constraint.

# Examples
```julia
probability_cut(model, π_s, P)
```
"""
function probability_cut(model::Model, π_s::PathProbabilityVariables, P::AbstractPathProbability)
    # Add the constraints only once
    ϵ = minimum(P(s) for s in keys(π_s))
    flag = false
    function probability_cut(cb_data)
        flag && return
        πsum = sum(callback_value(cb_data, π) for π in values(π_s))
        if !isapprox(πsum, 1.0, atol=ϵ)
            con = @build_constraint(sum(values(π_s)) == 1.0)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), probability_cut)
end

"""Adds a active paths cut to the model as a lazy constraint.

# Examples
```julia
atol = 0.9  # Tolerance to trigger the creation of the lazy cut
active_paths_cut(model, π_s, S, P; atol=atol)
```
"""
function active_paths_cut(model::Model, π_s::PathProbabilityVariables, S::States, P::AbstractPathProbability; atol::Float64 = 0.9)
    all_active_states = all(all((!).(iszero.(x))) for x in P.X)
    if !all_active_states
        throw(DomainError("Cannot use active paths cut if all states are not active."))
    end
    ϵ = minimum(P(s) for s in keys(π_s))
    num_compatible_paths = prod(S[c.j] for c in P.C)
    # Add the constraints only once
    flag = false
    function active_paths_cut(cb_data)
        flag && return
        πnum = sum(callback_value(cb_data, π) ≥ ϵ for π in values(π_s))
        if !isapprox(πnum, num_compatible_paths, atol = atol)
            num_active_paths = @expression(model, sum(π / P(s) for (s, π) in π_s))
            con = @build_constraint(num_active_paths == num_compatible_paths)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), active_paths_cut)
end


# --- Objective Functions ---

"""Positive affine transformation of path utility. Always evaluates positive values.

# Examples
```julia-repl
julia> U⁺ = PositivePathUtility(S, U)
julia> all(U⁺(s) > 0 for s in paths(S))
true
```
"""
struct PositivePathUtility <: AbstractPathUtility
    U::AbstractPathUtility
    min::Float64
    function PositivePathUtility(S::States, U::AbstractPathUtility)
        u_min = minimum(U(s) for s in paths(S))
        new(U, u_min)
    end
end

(U::PositivePathUtility)(s::Path) = U.U(s) - U.min + 1

"""Create an expected value objective.

# Examples
```julia
EV = expected_value(model, π_s, U)
```
"""
function expected_value(model::Model, π_s::PathProbabilityVariables, U::AbstractPathUtility)
    @expression(model, sum(π * U(s) for (s, π) in π_s))
end

"""Create a conditional value-at-risk (CVaR) objective.

# Examples
```julia
α = 0.05  # Parameter such that 0 ≤ α ≤ 1
CVaR = conditional_value_at_risk(model, π_s, U, α)
```
"""
function conditional_value_at_risk(model::Model, π_s::PathProbabilityVariables{N}, U::AbstractPathUtility, α::Float64) where N
    if !(0 < α ≤ 1)
        throw(DomainError("α should be 0 < α ≤ 1"))
    end

    # Pre-computed parameters
    u = collect(Iterators.flatten(U(s) for s in keys(π_s)))
    u_sorted = sort(u)
    u_min = u_sorted[1]
    u_max = u_sorted[end]
    M = u_max - u_min
    u_diff = diff(u_sorted)
    ϵ = if isempty(u_diff) 0.0 else minimum(filter(!iszero, abs.(u_diff))) / 2 end

    # Variables and constraints
    η = @variable(model)
    @constraint(model, η ≥ u_min)
    @constraint(model, η ≤ u_max)
    ρ′_s = Dict{NTuple{N, Int}, VariableRef}()
    for (s, π) in π_s
        u_s = U(s)
        λ = @variable(model, binary=true)
        λ′ = @variable(model, binary=true)
        ρ = @variable(model)
        ρ′ = @variable(model)
        @constraint(model, η - u_s ≤ M * λ)
        @constraint(model, η - u_s ≥ (M + ϵ) * λ - M)
        @constraint(model, η - u_s ≤ (M + ϵ) * λ′ - ϵ)
        @constraint(model, η - u_s ≥ M * (λ′ - 1))
        @constraint(model, 0 ≤ ρ)
        @constraint(model, 0 ≤ ρ′)
        @constraint(model, ρ ≤ λ)
        @constraint(model, ρ′ ≤ λ′)
        @constraint(model, ρ ≤ ρ′)
        @constraint(model, ρ′ ≤ π)
        @constraint(model, π - (1 - λ) ≤ ρ)
        ρ′_s[s] = ρ′
    end
    @constraint(model, sum(values(ρ′_s)) == α)

    # Return CVaR as an expression
    CVaR = @expression(model, sum(ρ_bar * U(s) for (s, ρ_bar) in ρ′_s) / α)

    return CVaR
end


# --- Construct decision strategy from JuMP variables ---

"""Construct decision strategy from variable refs."""
function LocalDecisionStrategy(j::Node, z::Array{VariableRef})
    LocalDecisionStrategy(j, @. Int(round(value(z))))
end

"""Extract values for decision variables from solved decision model.

# Examples
```julia
Z = DecisionStrategy(z, D)
```
"""
function DecisionStrategy(z::DecisionVariables, D::Vector{DecisionNode})
    DecisionStrategy(D, [LocalDecisionStrategy(d.j, v) for (d, v) in zip(D, z)])
end
