using JuMP

struct InformationStructureVariables{N} <: AbstractDict{Tuple{Node,Node}, VariableRef}
    data::Dict{Tuple{Node,Node}, VariableRef}
end

function decision_variable(model::Model, S::States, d::Node, I_d::Vector{Node},K::Vector{Tuple{Node,Node}},augmented_states::Bool,binary::Bool, base_name::String="")
    # Create decision variables.
    dims = S[[I_d; d]]

    if augmented_states
        K_j = filter(x -> x[2] == d,K)
        # Add 1 to dimensions for each k \in K(j)
        for i in K_j
            indices = findall(x->x==i[1], I_d)
            for j in indices
                dims[j] = dims[j] +1
            end
        end
    end

    z_d = Array{VariableRef}(undef, dims...)
    for s in paths(dims)
        if binary 
            z_d[s...] = @variable(model, binary=true, base_name="$(base_name)_$(s)")
        else
            z_d[s...] = @variable(model, base_name="$(base_name)_$(s)")
            @constraint(model,0 ≤ z_d[s...] ≤ 1.0)
        end
    end

    if augmented_states
        pop!(dims)
        # Paths that contain a zero state
        augmented_paths = Iterators.filter(x -> x ∉ paths(S[I_d]), paths(dims))
        for s_I in paths(S[I_d])
            zero_extension = Iterators.filter(s -> all((s_I.==s) .| (s .== (dims .+ 1))),augmented_paths)
            @constraint(model, sum(z_d[s_I..., s_d] + sum(z_d[s..., s_d] for s in zero_extension) for s_d in 1:S[d])  == 1)
        end
    else
        for s_I in paths(S[I_d])
            @constraint(model, sum(z_d[s_I..., s_d] for s_d in 1:S[d]) == 1)
        end
    end
    return z_d
end

struct DecisionVariables
    D::Vector{Node}
    I_d::Vector{Vector{Node}}
    z::Vector{<:Array{VariableRef}}
end

"""
    DecisionVariables(model::Model,  diagram::InfluenceDiagram; names::Bool=false, name::String="z", binary::Bool = true, augmented_states::Bool = false)

Create decision variables and constraints.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `names::Bool`: Use names or have JuMP variables be anonymous.
- `name::String`: Prefix for predefined decision variable naming convention.
- `binary::Bool`: If true, returns decision variables with binarity constraint. If false, returns decision variables constrained to interval [0,1]
- `augmented_states::Bool`: Must be true if `AugmentedStateVariables` are used.


# Examples
```julia
z = DecisionVariables(model, diagram)
```
"""
function DecisionVariables(model::Model, diagram::InfluenceDiagram; names::Bool=false, name::String="z", binary::Bool = true, augmented_states::Bool = false)
    DecisionVariables(diagram.D, diagram.I_j[diagram.D], [decision_variable(model, diagram.S, d, I_d,diagram.K,augmented_states, binary, (names ? "$(name)_$(d)" : "")) for (d, I_d) in zip(diagram.D, diagram.I_j[diagram.D])])
end

function is_forbidden(s::Path, forbidden_paths::Vector{ForbiddenPath})
    return !all(s[k]∉v for (k, v) in forbidden_paths)
end


function path_compatibility_variable(model::Model, p_s::Float64,is_one::Bool=false, base_name::String="")
    # Create a path compatiblity variable
    x = @variable(model, base_name=base_name)

    # Constraint on the lower and upper bounds.
     if is_one
        @constraint(model, 0 ≤ x ≤ 1)
    else
        @constraint(model, 0 ≤ x ≤ p_s)
    end


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


function decision_strategy_constraint(model::Model, S::States, d::Node, I_d::Vector{Node}, D::Vector{Node}, z::Array{VariableRef}, x_s::PathCompatibilityVariables, upper_bound::Bool, K::Vector{Tuple{Node,Node}}, augmented_states::Bool)

    # states of nodes in information structure (s_d | s_I(d))
    dims = S[[I_d; d]]

    # Theoretical upper bound based on number of paths with information structure (s_d | s_I(d)) divided by number of possible decision strategies in other decision nodes
    other_decisions = filter(j -> all(j != d_set for d_set in [I_d; d]), D)
    theoretical_ub = prod(S)/prod(dims)/ prod(S[other_decisions])

    # paths that have a corresponding path compatibility variable
    existing_paths = keys(x_s)

    if augmented_states 
        K_j = map(x -> x[1] , filter(x -> x[2] == d,K))
        for i in K_j
            indices = findall(x->x==i, I_d)
            for j in indices
                dims[j] = dims[j] +1
            end
        end
        augmented_paths = Iterators.filter(x -> x ∉ paths(S[[I_d; d]]), paths(dims))
    end

    for s_d_s_Id in paths(S[[I_d; d]]) # iterate through all information states and states of d
        # paths with (s_d | s_I(d)) information structure
        feasible_paths = filter(s -> s[[I_d; d]] == s_d_s_Id, existing_paths)
        if upper_bound
            if augmented_states
                feasible_augmented_paths = Iterators.filter(s -> all((s_d_s_Id.==s) .| (s .== (dims .+ 1))),augmented_paths)
                @constraint(model, sum(get(x_s, s, 0) for s in feasible paths)  ≤ (z[s_d_s_Id...] + sum(z[s...] for s in feasible_augmented_paths)) * min(length(feasible_paths), theoretical_ub))

            else
                @constraint(model, sum(get(x_s, s, 0) for s in feasible paths) ≤ z[s_d_s_Id...] * min(length(feasible_paths), theoretical_ub))
            end
        else
            if augmented_states
                feasible_augmented_paths = Iterators.filter(s -> all((s_d_s_Id.==s) .| (s .== (dims .+ 1))),augmented_paths)
                for s in feasible_paths
                    @constraint(model, get(x_s, s, 0)<= (z[s_d_s_Id...] + sum(z[s...] for s in feasible_augmented_paths)))
                end
            else
                for s in feasible_paths
                    @constraint(model, get(x_s, s, 0)<= z[s_d_s_Id...])
                end
            end
        end
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
    is_one::Bool = false,
    names::Bool=false,
    name::String="x",
    forbidden_paths::Vector{ForbiddenPath}=ForbiddenPath[],
    fixed::FixedPath=Dict{Node, State}(),
    probability_cut::Bool=true,
    probability_scale_factor::Float64=1.0,
    upper_bound::Bool = false,
    augmented_states::Bool = false)

    if probability_scale_factor ≤ 0
        throw(DomainError("The probability_scale_factor must be greater than 0."))
    end

    if !isempty(forbidden_paths)
        @warn("Forbidden paths is still an experimental feature.")
    end

    # Create path compatibility variable for each effective path.
    N = length(diagram.S)
    variables_x_s = Dict{Path{N}, VariableRef}(
        s => path_compatibility_variable(model, diagram.P(s),is_one, (names ? "$(name)$(s)" : ""))
        for s in paths(diagram.S, fixed)
        if !iszero(diagram.P(s)) && !is_forbidden(s, forbidden_paths)
    )

    x_s = PathCompatibilityVariables{N}(variables_x_s)

    # Add decision strategy constraints for each decision node
    for (d, z_d) in zip(z.D, z.z)
        decision_strategy_constraint(model, diagram.S, d, diagram.I_j[d], z.D, z_d, x_s,upper_bound,diagram.K,augmented_states)
    end

    if probability_cut
        @constraint(model, sum(x * diagram.P(s) * probability_scale_factor for (s, x) in x_s) == 1.0 * probability_scale_factor)
    end

    x_s
end

"""
    InformationConstraintVariables(model::Model,
        diagram::InfluenceDiagram,
        z::DecisionVariables,
        x_s::PathCompatibilityVariables;
        names::Bool=false,
        name::String="x")

Create information structure constraints for path probabilities.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `z::DecisionVariables`: Decision variables from `DecisionVariables` function.
- `x_s::PathCompatibilityVariables`: Path compatibility variables from `PathCompatibilityVariables` function.
- `names::Bool`: Use names or have JuMP variables be anonymous.
- `name::String`: Prefix for predefined decision variable naming convention.

# Examples
```julia
    x_x = InformationConstraintVariables(model, diagram, z, x_s, names=true, name="x")
```
"""

function InformationConstraintVariables(model::Model,
    diagram::InfluenceDiagram,
    z::DecisionVariables,
    x_s::PathCompatibilityVariables;
    names::Bool=false,
    name::String="x")


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
DecisionConstraintVariables(model::Model,
        diagram::InfluenceDiagram,
        z::DecisionVariables,
        x_s::PathCompatibilityVariables;
        names::Bool=false,
        name::String="x")

Create information structure constraints for local decisions.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `z::DecisionVariables`: Decision variables from `DecisionVariables` function.
- `x_s::PathCompatibilityVariables`: Path compatibility variables from `PathCompatibilityVariables` function.
- `names::Bool`: Use names or have JuMP variables be anonymous.
- `name::String`: Prefix for predefined decision variable naming convention.

# Examples
```julia
    x_x = DecisionConstraintVariables(model, diagram, z, x_s, names=true, name="x")
```
"""

function DecisionConstraintVariables(model::Model,
    diagram::InfluenceDiagram,
    z::DecisionVariables,
    x_s::PathCompatibilityVariables;
    names::Bool=false,
    name::String="x",
    set_sharing_to_one::Bool=false,
    forbidden_paths::Vector{ForbiddenPath}=ForbiddenPath[],
    fixed::FixedPath=Dict{Node, State}(),
    probability_cut::Bool=true,
    probability_scale_factor::Float64=1.0)


    # Create path compatibility variable for each effective path.
    N = length(diagram.S)
    variables_x = Dict{Tuple{Node,Node}, VariableRef}(
        s => information_structure_variable(model, (names ? "$(name)$(s)" : ""),set_sharing_to_one)
        for s in diagram.K
    )


    # Add information constraints for each decision node
    for (d, z_d) in zip(z.D, z.z)
        decision_path_constraints(model, diagram.S, d, diagram.I_j[d], z_d, x_s, diagram.K, variables_x)
    end
    return variables_x
end

function decision_path_constraints(model::Model, S::States, d::Node, I_d::Vector{Node}, z::Array{VariableRef}, x_s::PathCompatibilityVariables, K::Vector{Tuple{Node,Node}}, x_x::Dict{Tuple{Node,Node},VariableRef})
    # states of nodes in information structure (s_d | s_I(d))
    for k in filter(tup -> tup[2] == d, K)
        nodes = [I_d;d]
        k_index = findall(x -> x == k[1], nodes)
        Id_without_k = findall(x -> x != k[1], I_d)
        dims = S[[I_d; d]]

        for s_d_s_Id in paths(dims) # iterate through all information states and states of d
            # paths with (s_d | s_I(d)) information structure
            for s_d_s_k in paths(dims)
                if s_d_s_k[first(k_index)] != s_d_s_Id[first(k_index)] && s_d_s_k[Id_without_k] == s_d_s_Id[Id_without_k] && last(s_d_s_k) == last(s_d_s_Id)
                    @constraint(model, z[s_d_s_k...]  >=   z[s_d_s_Id...] - x_x[k])
                end
            end 
        end
    end
end

"""
    AugmentedStateVariables(model::Model,
        diagram::InfluenceDiagram,
        z::DecisionVariables,
        x_s::PathCompatibilityVariables;
        names::Bool=false,
        name::String="x")

Create information structure constraints for augmented states.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `z::DecisionVariables`: Decision variables from `DecisionVariables` function.
- `x_s::PathCompatibilityVariables`: Path compatibility variables from `PathCompatibilityVariables` function.
- `names::Bool`: Use names or have JuMP variables be anonymous.
- `name::String`: Prefix for predefined decision variable naming convention.

# Examples
```julia
    x_x = InformationConstraintVariables(model, diagram, z, x_s, names=true, name="x")
```
"""

function AugmentedStateVariables(model::Model,
    diagram::InfluenceDiagram,
    z::DecisionVariables,
    x_s::PathCompatibilityVariables;
    names::Bool=false,
    name::String="x")


    # Create path compatibility variable for each effective path.
    N = length(diagram.S)
    variables_x = Dict{Tuple{Node,Node}, VariableRef}(
        s => information_structure_variable(model, (names ? "$(name)$(s)" : ""))
        for s in diagram.K
    )


    # Add information constraints for each decision node
    for (d, z_d) in zip(z.D, z.z)
        augmented_state_constraints(model, diagram.S, d, diagram.I_j[d], z_d, x_s, diagram.K,variables_x)
    end
    return variables_x
end

function augmented_state_constraints(model::Model, S::States, d::Node, I_d::Vector{Node}, z::Array{VariableRef}, x_s::PathCompatibilityVariables, K::Vector{Tuple{Node,Node}}, x_x::Dict{Tuple{Node,Node},VariableRef})

    # states of nodes in information structure (s_d | s_I(d))
    dims = S[[I_d; d]]
    dims_3 = S[[I_d; d]]
    # paths that have a corresponding path compatibility variable
    existing_paths = paths(dims)
    K_j = map(x -> x[1] , filter(x -> x[2] == d,K))
    for i in K_j
        indices = findall(x->x==i, I_d)
        for j in indices
            dims_3[j] = dims_3[j] +1
        end
    end
    augmented_paths = Iterators.filter(x -> x ∉ existing_paths, paths(dims_3))

    for k in filter(tup -> tup[2] == d, K)
        indices = findall(y -> y == k[1] ,I_d)
        dimensions = dims[indices[1]]
        zero = Iterators.filter(x -> x[indices[1]] == dimensions + 1, augmented_paths)
        non_zero = Iterators.filter(x -> x[indices[1]] < dimensions + 1, paths(dims_3))
        @constraint(model,sum(z[s...] for s in non_zero) <= length(paths(dims_3))*x_x[k])
        @constraint(model,sum(z[s...] for s in zero) <= length(paths(dims_3))*(1-x_x[k]))
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
        x_s::PathCompatibilityVariables,
        x_x::Dict{Tuple{Node,Node},VariableRef})

Create an expected value objective.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `x_s::PathCompatibilityVariables`: Path compatibility variables.
- `x_x::InformationStructureVariables`: Information structure variables.

# Examples
```julia
EV = expected_value(model, diagram, x_s,x_x)
```
"""
function expected_value(model::Model,
    diagram::InfluenceDiagram,
    x_s::PathCompatibilityVariables;
    x_x::Dict{Tuple{Node,Node},VariableRef} = Dict{Tuple{Node,Node},VariableRef}())
    @expression(model, sum(x * diagram.U(s, diagram.translation)  for (s, x) in x_s) - sum(diagram.Cs[k] * x for (k,x) in x_x )+ sum(0.000001 * x for (k,x) in x_x ))
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
