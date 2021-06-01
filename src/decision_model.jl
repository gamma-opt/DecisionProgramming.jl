using JuMP

function decision_variable(model::Model, S::States, d::DecisionNode, base_name::String="")
    # Create decision variables.
    dims = S[[d.I_j; d.j]]
    z_j = Array{VariableRef}(undef, dims...)
    for s in paths(dims)
        z_j[s...] = @variable(model, binary=true, base_name=base_name)
    end
    # Constraints to one decision per decision strategy.
    for s_I in paths(S[d.I_j])
        @constraint(model, sum(z_j[s_I..., s_j] for s_j in 1:S[d.j]) == 1)
    end
    return z_j
end

struct DecisionVariables
    D::Vector{DecisionNode}
    z::Vector{<:Array{VariableRef}}
end

"""Create decision variables and constraints.

# Examples
```julia
z = DecisionVariables(model, S, D)
```
"""
function DecisionVariables(model::Model, S::States, D::Vector{DecisionNode}; names::Bool=false, name::String="z")
    DecisionVariables(D, [decision_variable(model, S, d, (names ? "$(name)_$(d.j)$(s)" : "")) for d in D])
end

function is_forbidden(s::Path, forbidden_paths::Vector{ForbiddenPath})
    return !all(s[k]∉v for (k, v) in forbidden_paths)
end

function path_probability_variable(model::Model, z::DecisionVariables, s::Path, P::AbstractPathProbability, hard_lower_bound::Bool, forbidden::Bool, base_name::String="", probability_scale_factor::Int64)
    # Create a path probability variable
    π = @variable(model, base_name=base_name)

    # Soft constraint on the lower bound.
    @constraint(model, π ≥ 0)

    if !forbidden

        # Hard constraint on the upper bound.
        @constraint(model, π ≤ P(s) * probability_scale_factor)

        # Constraints the path probability to zero if the path is
        # incompatible with the decision strategy.
        for (d, z_j) in zip(z.D, z.z)
            @constraint(model, π ≤ z_j[s[[d.I_j; d.j]]...] * probability_scale_factor)
        end
    else
        # Path is forbidden, probability must be zero
        @constraint(model, π ≤ 0)
    end

    # Hard constraint on the lower bound.
    if hard_lower_bound
        n_z = @expression(model, sum(z_j[s[[d.I_j; d.j]]...] for (d, z_j) in zip(z.D, z.z)))
        @constraint(model, π ≥ (P(s) + n_z - length(z.D)) * probability_scale_factor)
    end

    return π
end

struct PathProbabilityVariables{N} <: AbstractDict{Path{N}, VariableRef}
    data::Dict{Path{N}, VariableRef}
end

Base.getindex(π_s::PathProbabilityVariables, key) = getindex(π_s.data, key)
Base.get(π_s::PathProbabilityVariables, key, default) = get(π_s.data, key, default)
Base.keys(π_s::PathProbabilityVariables) = keys(π_s.data)
Base.values(π_s::PathProbabilityVariables) = values(π_s.data)
Base.pairs(π_s::PathProbabilityVariables) = pairs(π_s.data)
Base.iterate(π_s::PathProbabilityVariables) = iterate(π_s.data)
Base.iterate(π_s::PathProbabilityVariables, i) = iterate(π_s.data, i)

"""Create path probability variables and constraints.

# Examples
```julia
π_s = PathProbabilityVariables(model, z, S, P)
π_s = PathProbabilityVariables(model, z, S, P; hard_lower_bound=false))
```
"""
function PathProbabilityVariables(model::Model, z::DecisionVariables, S::States, P::AbstractPathProbability; hard_lower_bound::Bool=true, names::Bool=false, name::String="π_s", forbidden_paths::Vector{ForbiddenPath}=ForbiddenPath[], fixed::Dict{Node, State}=Dict{Node, State}(), probability_scale_factor::Int64=1)
    if !isempty(forbidden_paths)
        @warn("Forbidden paths is still an experimental feature.")
    end

    # Create path probability variable for each effective path.
    N = length(S)
    π_s = Dict{Path{N}, VariableRef}(
        s => path_probability_variable(model, z, s, P, hard_lower_bound, is_forbidden(s, forbidden_paths), (names ? "$(name)$(s)" : ""), probability_scale_factor)
        for s in paths(S, fixed)
        if !iszero(P(s))
    )
    PathProbabilityVariables{N}(π_s)
end

"""Adds a probability cut to the model as a lazy constraint.

# Examples
```julia
probability_cut(model, π_s, P)
```
"""
function probability_cut(model::Model, π_s::PathProbabilityVariables, P::AbstractPathProbability; probability_scale_factor::Int64=1)
    # Add the constraints only once
    ϵ = minimum(P(s) for s in keys(π_s))
    flag = false
    function probability_cut(cb_data)
        flag && return
        πsum = sum(callback_value(cb_data, π) for π in values(π_s))
        if !isapprox(πsum, 1.0 * probability_scale_factor, atol=ϵ)
            con = @build_constraint(sum(values(π_s)) == 1.0 * probability_scale_factor)
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
function active_paths_cut(model::Model, π_s::PathProbabilityVariables, S::States, P::AbstractPathProbability; atol::Float64 = 0.9, probability_scale_factor::Int64=1)
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
            num_active_paths = @expression(model, sum(π / (P(s) * probability_scale_factor) for (s, π) in π_s))
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
function expected_value(model::Model, π_s::PathProbabilityVariables, U::AbstractPathUtility; probability_scale_factor::In64=1)
    @expression(model, sum(π / probability_scale_factor * U(s) for (s, π) in π_s))
end

"""Create a conditional value-at-risk (CVaR) objective.

# Examples
```julia
α = 0.05  # Parameter such that 0 ≤ α ≤ 1
CVaR = conditional_value_at_risk(model, π_s, U, α)
```
"""
function conditional_value_at_risk(model::Model, π_s::PathProbabilityVariables{N}, U::AbstractPathUtility, α::Float64; probability_scale_factor::Int64=1) where N
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
    ρ′_s = Dict{Path{N}, VariableRef}()
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
        @constraint(model, ρ ≤ λ * probability_scale_factor)
        @constraint(model, ρ′ ≤ λ′* probability_scale_factor)
        @constraint(model, ρ ≤ ρ′)
        @constraint(model, ρ′ ≤ π)
        @constraint(model, π - (1 - λ)* probability_scale_factor ≤ ρ)
        ρ′_s[s] = ρ′
    end
    @constraint(model, sum(values(ρ′_s)) == α * probability_scale_factor)

    # Return CVaR as an expression
    CVaR = @expression(model, sum(ρ_bar * U(s) for (s, ρ_bar) in ρ′_s) / (α * probability_scale_factor))

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
Z = DecisionStrategy(z)
```
"""
function DecisionStrategy(z::DecisionVariables)
    DecisionStrategy(z.D, [LocalDecisionStrategy(d.j, v) for (d, v) in zip(z.D, z.z)])
end
