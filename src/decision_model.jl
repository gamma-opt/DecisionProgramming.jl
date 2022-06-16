using JuMP

struct InformationStructureVariables{N} <: AbstractDict{Tuple{Node,Node}, VariableRef}
    data::Dict{Tuple{Node,Node}, VariableRef}
end

function decision_variable(model::Model, S::States, d::Node, I_d::Vector{Node}, base_name::String="")
    # Create decision variables.
    dims = S[[I_d; d]]
    z_d = Array{VariableRef}(undef, dims...)
    for s in paths(dims)
        z_d[s...] = @variable(model, binary=true, base_name=base_name)
    end
    # Constraints to one decision per decision strategy.
    for s_I in paths(S[I_d])
        @constraint(model, sum(z_d[s_I..., s_d] for s_d in 1:S[d]) == 1)
    end
    return z_d
end

struct DecisionVariables
    D::Vector{Node}
    I_d::Vector{Vector{Node}}
    z::Vector{<:Array{VariableRef}}
end

"""
    DecisionVariables(model::Model,  diagram::InfluenceDiagram; names::Bool=false, name::String="z")

Create decision variables and constraints.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `names::Bool`: Use names or have JuMP variables be anonymous.
- `name::String`: Prefix for predefined decision variable naming convention.


# Examples
```julia
z = DecisionVariables(model, diagram)
```
"""
function DecisionVariables(model::Model, diagram::InfluenceDiagram; names::Bool=false, name::String="z")
    DecisionVariables(diagram.D, diagram.I_j[diagram.D], [decision_variable(model, diagram.S, d, I_d, (names ? "$(name)_$(d.j)$(s)" : "")) for (d, I_d) in zip(diagram.D, diagram.I_j[diagram.D])])
end

function is_forbidden(s::Path, forbidden_paths::Vector{ForbiddenPath})
    return !all(s[k]∉v for (k, v) in forbidden_paths)
end


function path_compatibility_variable(model::Model, base_name::String="")
    # Create a path compatiblity variable
    x = @variable(model, base_name=base_name)

    # Constraint on the lower and upper bounds.
    @constraint(model, 0 ≤ x ≤ 1.0)

    return x
end

function information_structure_variable(model::Model, base_name::String="",is_one::Bool=false)
    # Create a path compatiblity variable
    x = @variable(model, base_name=base_name, binary=true)
    if is_one
        @constraint(model, 0.99 ≤ x ≤ 1.01)
    end
    return x
end

struct PathCompatibilityVariables{N} <: AbstractDict{Path{N}, VariableRef}
    data::Dict{Path{N}, VariableRef}
end

Base.length(x_s::PathCompatibilityVariables) = length(x_s.data)
Base.getindex(x_s::PathCompatibilityVariables, key) = getindex(x_s.data, key)
Base.get(x_s::PathCompatibilityVariables, key, default) = get(x_s.data, key, default)
Base.keys(x_s::PathCompatibilityVariables) = keys(x_s.data)
Base.values(x_s::PathCompatibilityVariables) = values(x_s.data)
Base.pairs(x_s::PathCompatibilityVariables) = pairs(x_s.data)
Base.iterate(x_s::PathCompatibilityVariables) = iterate(x_s.data)
Base.iterate(x_s::PathCompatibilityVariables, i) = iterate(x_s.data, i)


function decision_strategy_constraint(model::Model, S::States, d::Node, I_d::Vector{Node}, D::Vector{Node}, z::Array{VariableRef}, x_s::PathCompatibilityVariables)

    # states of nodes in information structure (s_d | s_I(d))
    dims = S[[I_d; d]]

    # Theoretical upper bound based on number of paths with information structure (s_d | s_I(d)) divided by number of possible decision strategies in other decision nodes
    other_decisions = filter(j -> all(j != d_set for d_set in [I_d; d]), D)
    theoretical_ub = prod(S)/prod(dims)/ prod(S[other_decisions])

    # paths that have a corresponding path compatibility variable
    existing_paths = keys(x_s)

    for s_d_s_Id in paths(dims) # iterate through all information states and states of d
        # paths with (s_d | s_I(d)) information structure
        feasible_paths = filter(s -> s[[I_d; d]] == s_d_s_Id, existing_paths)

        @constraint(model, sum(get(x_s, s, 0) for s in feasible_paths) ≤ z[s_d_s_Id...] * min(length(feasible_paths), theoretical_ub))
    end
end

"""
    PathCompatibilityVariables(model::Model,
        diagram::InfluenceDiagram,
        z::DecisionVariables;
        names::Bool=false,
        name::String="x",
        forbidden_paths::Vector{ForbiddenPath}=ForbiddenPath[],
        fixed::FixedPath=Dict{Node, State}(),
        probability_cut::Bool=true,
        probability_scale_factor::Float64=1.0)

Create path compatibility variables and constraints.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `z::DecisionVariables`: Decision variables from `DecisionVariables` function.
- `names::Bool`: Use names or have JuMP variables be anonymous.
- `name::String`: Prefix for predefined decision variable naming convention.
- `forbidden_paths::Vector{ForbiddenPath}`: The forbidden subpath structures.
    Path compatibility variables will not be generated for paths that include
    forbidden subpaths.
- `fixed::FixedPath`: Path compatibility variable will not be generated
    for paths which do not include these fixed subpaths.
- `probability_cut` Includes probability cut constraint in the optimisation model.
- `probability_scale_factor::Float64`: Adjusts conditional value at risk model to
   be compatible with the expected value expression if the probabilities were scaled there.

# Examples
```julia
x_s = PathCompatibilityVariables(model, diagram; probability_cut = false)
```
"""
function PathCompatibilityVariables(model::Model,
    diagram::InfluenceDiagram,
    z::DecisionVariables;
    names::Bool=false,
    name::String="x",
    forbidden_paths::Vector{ForbiddenPath}=ForbiddenPath[],
    fixed::FixedPath=Dict{Node, State}(),
    probability_cut::Bool=true,
    probability_scale_factor::Float64=1.0)

    if probability_scale_factor ≤ 0
        throw(DomainError("The probability_scale_factor must be greater than 0."))
    end

    if !isempty(forbidden_paths)
        @warn("Forbidden paths is still an experimental feature.")
    end

    # Create path compatibility variable for each effective path.
    N = length(diagram.S)
    variables_x_s = Dict{Path{N}, VariableRef}(
        s => path_compatibility_variable(model, (names ? "$(name)$(s)" : ""))
        for s in paths(diagram.S, fixed)
        if !iszero(diagram.P(s)) && !is_forbidden(s, forbidden_paths)
    )

    x_s = PathCompatibilityVariables{N}(variables_x_s)

    # Add decision strategy constraints for each decision node
    for (d, z_d) in zip(z.D, z.z)
        decision_strategy_constraint(model, diagram.S, d, diagram.I_j[d], z.D, z_d, x_s)
    end

    if probability_cut
        @constraint(model, sum(x * diagram.P(s) * probability_scale_factor for (s, x) in x_s) == 1.0 * probability_scale_factor)
    end

    x_s
end

function InformationConstraintVariables(model::Model,
    diagram::InfluenceDiagram,
    z::DecisionVariables,
    x_s::PathCompatibilityVariables;
    names::Bool=false,
    name::String="x",
    forbidden_paths::Vector{ForbiddenPath}=ForbiddenPath[],
    fixed::FixedPath=Dict{Node, State}(),
    probability_cut::Bool=true,
    probability_scale_factor::Float64=1.0)


    # Create information structure variable for each correspondig conditional edge.
    variables_x = Dict{Tuple{Node,Node}, VariableRef}(
        s => information_structure_variable(model, (names ? "$(name)$(s)" : ""))
        for s in diagram.K
    )


    # Add information constraints for each decision node
    for (d, z_d) in zip(z.D, z.z)
        information_constraints(model, diagram.S, d, diagram.I_j[d], z_d, x_s, diagram.K,variables_x)
    end
    return variables_x
end

function information_constraints(model::Model, S::States, d::Node, I_d::Vector{Node}, z::Array{VariableRef}, x_s::PathCompatibilityVariables, K::Vector{Tuple{Node,Node}}, x_x::Dict{Tuple{Node,Node},VariableRef})
    # states of nodes in information structure (s_d | s_I(d))
    for k in filter(tup -> tup[2] == d, K)
        nodes = [I_d;d]
        d_index = findall(x -> x == d, nodes)
        k_index = findall(x -> x == k[1], nodes)
        Id_index = findall(x -> x in I_d && x != k[1], nodes)
        Id_without_k = filter(x -> x != k[1], I_d)
        dims = S[[I_d; d]]

        # paths that have a corresponding path compatibility variable
        existing_paths = keys(x_s)

        for s_d_s_Id in paths(dims) # iterate through all information states and states of d
            # paths with (s_d | s_I(d)) information structure
            s_prime = filter(s -> s[d] != s_d_s_Id[first(d_index)] && s[Id_without_k] == s_d_s_Id[Id_index] && s[k[1]] != s_d_s_Id[first(k_index)], existing_paths)
            for s in s_prime
                @constraint(model, get(x_s, s, 0) <= 1 - z[s_d_s_Id...] + x_x[k])
            end
        end
    end
end

"""
    lazy_probability_cut(model::Model, diagram::InfluenceDiagram, x_s::PathCompatibilityVariables)

Add a probability cut to the model as a lazy constraint.

# Examples
```julia
lazy_probability_cut(model, diagram, x_s)
```

!!! note
    Remember to set lazy constraints on in the solver parameters, unless your solver does this automatically. Note that Gurobi does this automatically.

"""
function lazy_probability_cut(model::Model, diagram::InfluenceDiagram, x_s::PathCompatibilityVariables)
    # August 2021: The current implementation of JuMP doesn't allow multiple callback functions of the same type (e.g. lazy)
    # (see https://github.com/jump-dev/JuMP.jl/issues/2642)
    # What this means is that if you come up with a new lazy cut, you must replace this
    # function with a more general function (see discussion and solution in https://github.com/gamma-opt/DecisionProgramming.jl/issues/20)

    function probability_cut(cb_data)
        xsum = sum(callback_value(cb_data, x) * diagram.P(s) for (s, x) in x_s)
        if !isapprox(xsum, 1.0)
            con = @build_constraint(sum(x * diagram.P(s) for (s, x) in x_s) == 1.0)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), probability_cut)
end

"""
    expected_value(model::Model,
        diagram::InfluenceDiagram,
        x_s::PathCompatibilityVariables)

Create an expected value objective.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `x_s::PathCompatibilityVariables`: Path compatibility variables.

# Examples
```julia
EV = expected_value(model, diagram, x_s)
```
"""
function expected_value(model::Model,
    diagram::InfluenceDiagram,
    x_s::PathCompatibilityVariables,
    x_x::Dict{Tuple{Node,Node},VariableRef})
    @expression(model, sum(diagram.P(s) * x * diagram.U(s, diagram.translation)  for (s, x) in x_s) - sum(diagram.Cs[k] * x for (k,x) in x_x )+ sum(0.000001 * x for (k,x) in x_x ))
end

"""
    conditional_value_at_risk(model::Model,
        diagram,
        x_s::PathCompatibilityVariables{N},
        α::Float64;
        probability_scale_factor::Float64=1.0) where N

Create a conditional value-at-risk (CVaR) objective.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `x_s::PathCompatibilityVariables`: Path compatibility variables.
- `α::Float64`: Probability level at which conditional value-at-risk is optimised.
- `probability_scale_factor::Float64`: Adjusts conditional value at risk model to
   be compatible with the expected value expression if the probabilities were scaled there.



# Examples
```julia
α = 0.05  # Parameter such that 0 ≤ α ≤ 1
CVaR = conditional_value_at_risk(model, x_s, U, P, α)
CVaR = conditional_value_at_risk(model, x_s, U, P, α; probability_scale_factor = 10.0)
```
"""
function conditional_value_at_risk(model::Model,
    diagram::InfluenceDiagram,
    x_s::PathCompatibilityVariables{N},
    α::Float64;
    probability_scale_factor::Float64=1.0) where N

    if probability_scale_factor ≤ 0
        throw(DomainError("The probability_scale_factor must be greater than 0."))
    end
    if !(0 < α ≤ 1)
        throw(DomainError("α should be 0 < α ≤ 1"))
    end

    # Pre-computed parameters
    u = collect(Iterators.flatten(diagram.U(s, diagram.translation) for s in keys(x_s)))
    u_sorted = sort(u)
    u_min = u_sorted[1]
    u_max = u_sorted[end]
    M = u_max - u_min
    u_diff = diff(u_sorted)
    if isempty(filter(!iszero, u_diff))
        return u_min    # All utilities are the same, CVaR is equal to that constant utility value
    else
        ϵ = minimum(filter(!iszero, abs.(u_diff))) / 2 
    end

    # Variables and constraints
    η = @variable(model)
    @constraint(model, η ≥ u_min)
    @constraint(model, η ≤ u_max)
    ρ′_s = Dict{Path{N}, VariableRef}()
    for (s, x) in x_s
        u_s = diagram.U(s, diagram.translation)
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
        @constraint(model, ρ′ ≤ x * diagram.P(s) * probability_scale_factor)
        @constraint(model, (x * diagram.P(s) - (1 - λ))* probability_scale_factor ≤ ρ)
        ρ′_s[s] = ρ′
    end
    @constraint(model, sum(values(ρ′_s)) == α * probability_scale_factor)

    # Return CVaR as an expression
    CVaR = @expression(model, sum(ρ_bar * diagram.U(s, diagram.translation) for (s, ρ_bar) in ρ′_s) / (α * probability_scale_factor))

    return CVaR
end

# --- Construct decision strategy from JuMP variables ---

"""
    LocalDecisionStrategy(j::Node, z::Array{VariableRef})

Construct decision strategy from variable refs.
"""
function LocalDecisionStrategy(d::Node, z::Array{VariableRef})
    LocalDecisionStrategy(d, @. Int(round(value(z))))
end

"""
    DecisionStrategy(z::DecisionVariables)

Extract values for decision variables from solved decision model.

# Examples
```julia
Z = DecisionStrategy(z)
```
"""
function DecisionStrategy(z::DecisionVariables)
    DecisionStrategy(z.D, z.I_d, [LocalDecisionStrategy(d, z_var) for (d, z_var) in zip(z.D, z.z)])
end
