using JuMP

"""Positive affine transformation of path utility.

# Examples
```julia
U⁺ = PositivePathUtility(S, U)
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

"""Evaluate positive affine transformation of the path utility.

# Examples
```julia-repl
julia> all(U⁺(s) ≥ 1 for s in paths(S))
true
```
"""
(U::PositivePathUtility)(s::Path) = U.U(s) - U.min + 1

"""Create a multidimensional array of JuMP variables.

# Examples
```julia
model = Model()
v1 = variables(model, [2, 3, 2])
v2 = variables(model, [2, 3, 2]; binary=true)
```
"""
function variables(model::Model, dims::AbstractVector{Int}; binary::Bool=false, names::Union{Nothing, Array{String}}=nothing)
    v = Array{VariableRef}(undef, dims...)
    for i in eachindex(v)
        name = if names === nothing; "" else names[i] end
        v[i] = @variable(model, binary=binary, base_name = name)
    end
    return v
end

"""Create decision variables and constraints.

# Examples
```julia
z = decision_variables(model, S, D)
```
"""
function decision_variables(model::Model, S::States, D::Vector{DecisionNode}; names::Bool=false, base_name::String="z")
    z = Vector{Array{VariableRef}}()
    for d in D
        # Create decision variables.
        dims = S[[d.I_j; d.j]]
        z_names = if names; ["$(base_name)_$(d.j)$(s)" for s in paths(dims)] else nothing end
        z_j = variables(model, dims; binary=true, names=z_names)

        # Constraints to one decision per decision strategy.
        for s_I in paths(S[d.I_j])
            @constraint(model, sum(z_j[s_I..., s_j] for s_j in 1:S[d.j]) == 1)
        end

        push!(z, z_j)
    end
    return z
end

"""Create path probability variables and constraints.

# Examples
```julia
π_s = path_probability_variables(model, z, S, D, P)
π_s = path_probability_variables(model, z, S, D, P; hard_lower_bound=false))
```
"""
function path_probability_variables(model::Model, z::Vector{<:Array{VariableRef}}, S::States, D::Vector{DecisionNode}, P::AbstractPathProbability; hard_lower_bound::Bool=true, names::Bool=false, base_name::String="π_s")
    # Create path probability variables.
    π_names = if names; ["$(base_name)$(s)" for s in paths(S)] else nothing end
    π_s = variables(model, S; names=π_names)

    # Create constraints for each variable.
    for s in paths(S)
        if iszero(P(s))
            # If the upper bound is zero, we fix the value to zero.
            fix(π_s[s...], 0)
        else
            # Soft constraint on the lower bound.
            @constraint(model, π_s[s...] ≥ 0)

            # Hard constraint on the upper bound.
            @constraint(model, π_s[s...] ≤ P(s))

            # Constraints the path probability to zero if the path is
            # incompatible with the decision strategy.
            for (d, z_j) in zip(D, z)
                @constraint(model, π_s[s...] ≤ z_j[s[[d.I_j; d.j]]...])
            end

            # Hard constraint on the lower bound.
            if hard_lower_bound
                @constraint(model,
                    π_s[s...] ≥ P(s) + sum(z_j[s[[d.I_j; d.j]]...] for (d, z_j) in zip(D, z)) - length(D))
            end
        end
    end
    return π_s
end

"""Adds a probability cut to the model as a lazy constraint.

# Examples
```julia
probability_cut(model, π_s, S, P)
```
"""
function probability_cut(model::Model, π_s::Array{VariableRef}, S::States, P::AbstractPathProbability)
    # Add the constraints only once
    ϵ = minimum(P(s) for s in paths(S))
    flag = false
    function probability_cut(cb_data)
        flag && return
        πsum = sum(callback_value(cb_data, π_s[s]) for s in eachindex(π_s))
        if !isapprox(πsum, 1.0, atol=ϵ)
            con = @build_constraint(sum(π_s) == 1.0)
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
function active_paths_cut(model::Model, π_s::Array{VariableRef}, S::States, P::AbstractPathProbability; atol::Float64 = 0.9)
    all_active_states = all(all((!).(iszero.(x))) for x in P.X)
    if !all_active_states
        throw(DomainError("Cannot use active paths cut if all states are not active."))
    end
    ϵ = minimum(P(s) for s in paths(S))
    num_compatible_paths = prod(S[c.j] for c in P.C)
    # Add the constraints only once
    flag = false
    function active_paths_cut(cb_data)
        flag && return
        πnum = sum(callback_value(cb_data, π_s[s]) ≥ ϵ for s in eachindex(π_s))
        if !isapprox(πnum, num_compatible_paths, atol = atol)
            num_active_paths = @expression(model, sum(π_s[s...] / P(s) for s in paths(S) if !iszero(P(s))))
            con = @build_constraint(num_active_paths == num_compatible_paths)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            flag = true
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), active_paths_cut)
end


# --- Objective Functions ---

"""Create an expected value objective.

# Examples
```julia
EV = expected_value(model, π_s, S, U)
```
"""
function expected_value(model::Model, π_s::Array{VariableRef}, S::States, U::AbstractPathUtility)
    @expression(model, sum(π_s[s...] * U(s) for s in paths(S)))
end

"""Create a conditional value-at-risk (CVaR) objective.

# Examples
```julia
α = 0.05  # Parameter such that 0 ≤ α ≤ 1
CVaR = conditional_value_at_risk(model, π_s, S, U, α)
```
"""
function conditional_value_at_risk(model::Model, π_s::Array{VariableRef}, S::States, U::AbstractPathUtility, α::Float64)
    if !(0 < α ≤ 1)
        throw(DomainError("α should be 0 < α ≤ 1"))
    end

    # Pre-computed parameters
    u = collect(Iterators.flatten(U(s) for s in paths(S)))
    u_sorted = sort(u)
    u_min = u_sorted[1]
    u_max = u_sorted[end]
    M = u_max - u_min
    u_diff = diff(u_sorted)
    ϵ = if isempty(u_diff) 0.0 else minimum(filter(!iszero, abs.(u_diff))) / 2 end

    # Variables
    η = @variable(model)
    λ = variables(model, S; binary=true)
    λ_bar = variables(model, S; binary=true)
    ρ = variables(model, S)
    ρ_bar = variables(model, S)

    # Constraints
    @constraint(model, η ≥ u_min)
    @constraint(model, η ≤ u_max)
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
        @constraint(model, ρ_bar[s...] ≤ π_s[s...])
        @constraint(model, π_s[s...] - (1 - λ[s...]) ≤ ρ[s...])
    end
    @constraint(model, sum(ρ_bar[s...] for s in paths(S)) == α)

    # Return CVaR as an expression
    CVaR = @expression(model, sum(ρ_bar[s...] * U(s) for s in paths(S)) / α)

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
function DecisionStrategy(z::Vector{<:Array{VariableRef}}, D::Vector{DecisionNode})
    DecisionStrategy(D, [LocalDecisionStrategy(d.j, v) for (d, v) in zip(D, z)])
end
