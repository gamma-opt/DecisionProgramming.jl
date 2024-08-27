using JuMP

function decision_variable(model::Model, S::States, d::Node, I_d::Vector{Node}, names::Bool, base_name::String="")
    # Create decision variables.
    dims = S[[I_d; d]]
    z_d = Array{VariableRef}(undef, dims...)
    for s in paths(dims)
        if names == true
            name = join([base_name, s...], "_")
            z_d[s...] = @variable(model, binary=true, base_name=name)
        else
            z_d[s...] = @variable(model, binary=true)
        end
    end
    # Constraints to one decision per decision strategy.
    for s_I in paths(S[I_d])
        @constraint(model, sum(z_d[s_I..., s_d] for s_d in 1:S[d]) == 1)
    end
    return z_d
end

struct DecisionVariable
    D::Name
    I_d::Vector{Name}
    z::Array{VariableRef}
end

"""
    DecisionVariables(model::Model,  diagram::InfluenceDiagram; names::Bool=true)

Create decision variables and constraints.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `names::Bool`: Use names or have JuMP variables be anonymous.


# Examples
```julia
z = DecisionVariables(model, diagram)
```
"""
function DecisionVariables(model::Model, diagram::InfluenceDiagram; names::Bool=true)
    decVars = OrderedDict{Name, DecisionVariable}()

    for (key, node) in diagram.D
        states = States(get_values(diagram.S))
        I_d = convert(Vector{Node}, indices_in_vector(diagram, diagram.D[key].I_j))
        base_name = names ? "$(diagram.D[key].name)" : ""

        decVars[key] = DecisionVariable(key, diagram.D[key].I_j, decision_variable(model, states, node.index, I_d, names, base_name)) 
    end

    return decVars
end

function is_forbidden(s::Path, forbidden_paths::Vector{ForbiddenPath})
    return !all(s[k]∉v for (k, v) in forbidden_paths)
end


function path_compatibility_variable(model::Model, base_name::String="")
    # Create a path compatiblity variable
    return @variable(model, base_name=base_name, lower_bound = 0, upper_bound = 1)
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
        z::OrderedDict{Name, DecisionVariable};
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
- `z::OrderedDict{Name, DecisionVariable}`: Ordered dictionary of decision variables.
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
x_s = PathCompatibilityVariables(model, diagram, z; probability_cut = false)
```
"""
function PathCompatibilityVariables(model::Model,
    diagram::InfluenceDiagram,
    z::OrderedDict{Name, DecisionVariable};
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
        for s in paths(get_values(diagram.S), fixed)
        if !iszero(diagram.P(s)) && !is_forbidden(s, forbidden_paths)
    )

    x_s = PathCompatibilityVariables{N}(variables_x_s)

    # Add decision strategy constraints for each decision node
    I_j_indices_result = I_j_indices(diagram, diagram.Nodes)
    z_indices = indices(diagram.D)
    z_z = [decision_node.z for decision_node in get_values(z)]

    for (d, z_d) in zip(z_indices, z_z)
        decision_strategy_constraint(model, States(get_values(diagram.S)), d, I_j_indices_result[d], z_indices, z_d, x_s)
    end

    if probability_cut
        cons = sum(x * diagram.P(s) * probability_scale_factor for (s, x) in x_s)
        @constraint(model, probability_cut, cons == 1.0 * probability_scale_factor)
    end

    return x_s
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
    x_s::PathCompatibilityVariables)

    diff_sign_utilities = false
    if minimum.(diagram.U.Y)[1]*maximum.(diagram.U.Y)[1] < 0.0
        diff_sign_utilities = true
    end

    if isnothing(constraint_by_name(model, "probability_cut")) && diff_sign_utilities
        throw(DomainError("The model contains both negative and positive utilities and no probability cut, which can lead to incorrect results. Probability cut constraint can be added using function PathCompatibilityVariables."))
    end

    @expression(model, sum(diagram.P(s) * x * diagram.U(s, diagram.translation) for (s, x) in x_s))
end

"""
    conditional_value_at_risk(model::Model,
        diagram::InfluenceDiagram,
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

    if isnothing(constraint_by_name(model, "probability_cut"))
        throw(DomainError("A probability cut constraint using PathCompatibilityVariables has to be created in order for decision programming path based CVaR to work."))
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
    LocalDecisionStrategy(d::Node, z::Array{VariableRef})

Construct decision strategy from variable refs.
"""
function LocalDecisionStrategy(d::Node, z::Array{VariableRef})
    LocalDecisionStrategy(d, @. Int(round(value(z))))
end

"""
    DecisionStrategy(diagram::InfluenceDiagram, z::OrderedDict{Name, DecisionVariable})

Extract values for decision variables from solved decision model.

# Examples
```julia
Z = DecisionStrategy(diagram, z)
```
"""
function DecisionStrategy(diagram::InfluenceDiagram, z::OrderedDict{Name, DecisionVariable})
    z_D = convert(Vector{Node}, indices_in_vector(diagram, get_keys(z)))
    z_I_d_Names = [decision_node.I_d for decision_node in get_values(z)]

    z_I_d_indices = [indices_in_vector(diagram, I_j) for I_j in z_I_d_Names]
    z_z = [decision_node.z for decision_node in get_values(z)]

    DecisionStrategy(z_D, z_I_d_indices, [LocalDecisionStrategy(d, z_var) for (d, z_var) in zip(z_D, z_z)])
end


# --- RJT MODEL ---


# Implements Algorithm 1 from Parmentier et al. (2020)
function ID_to_RJT(diagram::InfluenceDiagram)
    C_rjt = Dict{Name, Vector{Name}}()
    A_rjt = []
    names = get_keys(diagram.Nodes)
    for j in length(diagram.Nodes):-1:1
        C_j = copy(get_values(diagram.I_j)[j])
        push!(C_j, names[j])
        for a in A_rjt 
            if a[1] == names[j]
                push!(C_j, setdiff(C_rjt[a[2]], [a[2]])...)
            end
        end
        C_j = unique(C_j)
        C_j_aux = sort([(elem, findfirst(isequal(elem), names)) for elem in C_j], by = last)
        C_j = [C_j_tuple[1] for C_j_tuple in C_j_aux]
        C_rjt[names[j]] = C_j
        
        if length(C_rjt[names[j]]) > 1
            u = maximum([findfirst(isequal(name), names) for name in setdiff(C_j, [names[j]])])
            push!(A_rjt, (names[u], names[j]))
        end
    end
    return C_rjt, A_rjt
end


function add_variable(model::Model, states::Vector, name::Name, names::Bool)
    variable = @variable(model, [1:prod(length.(states))], lower_bound = 0)
    if names==true
        if !isempty(states) #if a variable is created (only exception should be the μ_bar variable for the root cluster)
            for (variable_i, states_i) in zip(variable, Iterators.product(states...))
                set_name(variable_i, "$name[$(join(states_i, ", "))]")
            end
        end
    end
    return Containers.DenseAxisArray(reshape(variable, length.(states)...), states...)
end


function μ_variable(model::Model, name::Name, S::OrderedDict{Name, Vector{Name}}, C_rjt::Dict{Name, Vector{Name}}, names::Bool)
    μ_statevars = add_variable(model, getindex.(Ref(S), C_rjt[name]), name, names::Bool)
    # Probability distributions μ sum to 1
    @constraint(model, sum(μ_statevars) == 1)
    return μ_statevars
end

function μ_bar_variable(model::Model, name::Name, S::OrderedDict{Name, Vector{Name}}, C_rjt::Dict{Name, Vector{Name}}, μ_statevars::Array{VariableRef}, names::Bool)
    μ_bar_statevars = add_variable(model, getindex.(Ref(S), setdiff(C_rjt[name], [name])), name, names::Bool)
    for index in CartesianIndices(μ_bar_statevars)
        # μ_bar defined as marginal distribution for μ-variables with one dimension marginalized out
        @constraint(model, μ_bar_statevars[index] .== dropdims(sum(μ_statevars, dims=findfirst(isequal(name), C_rjt[name])), dims=findfirst(isequal(name), C_rjt[name]))[index])
    end
    return μ_bar_statevars
end

struct μVariable
    node::Name
    statevars::Array{VariableRef}
end


function local_consistency_constraint(model::Model, arc::Tuple{Name, Name}, C_rjt::Dict{Name, Vector{Name}}, μVars::Dict{Name, μVariable})
    intersection = C_rjt[arc[1]] ∩ C_rjt[arc[2]]
    C1_minus_C2 = Tuple(setdiff(collect(1:length(C_rjt[arc[1]])), indexin(intersection, C_rjt[arc[1]])))
    C2_minus_C1 = Tuple(setdiff(collect(1:length(C_rjt[arc[2]])), indexin(intersection, C_rjt[arc[2]])))
    @constraint(model, 
        dropdims(sum(μVars[arc[1]].statevars, dims=C1_minus_C2), dims=C1_minus_C2) .== 
        dropdims(sum(μVars[arc[2]].statevars, dims=C2_minus_C1), dims=C2_minus_C1))
end


function factorization_constraints(model::Model, diagram::InfluenceDiagram, name::Name, μ_statevars::Array{VariableRef}, μ_bar_statevars::Array{VariableRef}, z::OrderedDict{Name, DecisionVariable})
    I_j_mapping_in_cluster = [findfirst(isequal(node), diagram.RJT.clusters[name]) for node in diagram.I_j[name]] # Map the information set to the variables in the cluster
    for index in CartesianIndices(μ_bar_statevars)
        for s_j in 1:length(diagram.States[name])
            if isa(diagram.Nodes[name], ChanceNode)
                # μ_{C_v} = μ_{\bar{C}_v}*p
                @constraint(model, μ_statevars[Tuple(index)...,s_j] == diagram.X[name][Tuple(index)[I_j_mapping_in_cluster]...,s_j]*μ_bar_statevars[index])
            elseif isa(diagram.Nodes[name], DecisionNode)
                # μ_{C_v} ≤ z
                @constraint(model, μ_statevars[Tuple(index)...,s_j] <= z[name].z[Tuple(index)[I_j_mapping_in_cluster]...,s_j])
            end
        end
    end
end


struct RJTVariables
    data::Dict{Name, μVariable}
end

"""
    RJTVariables(model, diagram, z)

Using the influence diagram and decision variables z from DecisionProgramming.jl, adds the 
variables and constraints of the corresponding RJT model.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `z::OrderedDict{Name, DecisionVariable}`: Decision variables from `DecisionVariables` function.

# Examples
```julia
μ_s = RJTVariables(model, diagram, z)
```
"""
function RJTVariables(model::Model, diagram::InfluenceDiagram, z::OrderedDict{Name, DecisionVariable}; names::Bool=true)
    # Get the RJT structure
    arcs, clusters = ID_to_RJT(diagram)
    diagram.RJT = RJT(arcs, clusters)

    # Variables corresponding to the nodes in the RJT
    μVars = Dict{Name, μVariable}()
    # Variables μ_{\bar{C}_v} = ∑_{x_v} μ_{C_v}
    μBarVars = Dict{Name, μVariable}()
    for name in union(keys(diagram.C), keys(diagram.D))
        μVars[name] = μVariable(name, μ_variable(model, name, diagram.States, diagram.RJT.clusters, names))
        μBarVars[name] = μVariable(name, μ_bar_variable(model, name, diagram.States, diagram.RJT.clusters, μVars[name].statevars, names))
    end

    # Enforcing local consistency between clusters, meaning that for a pair of adjacent clusters, 
    # the marginal distribution for nodes that are included in both, must be the same.
    for arc in diagram.RJT.arcs
        if !isa(diagram.Nodes[arc[2]], ValueNode)
            local_consistency_constraint(model, arc, diagram.RJT.clusters, μVars)
        end
    end

    # Add in the conditional probabilities and decision strategies
    # In our structure, value nodes are not stochastic. However, adding a chance node representing stochasticity before the value node is a possibility.
    for name in union(keys(diagram.C), keys(diagram.D))
        factorization_constraints(model, diagram, name, μVars[name].statevars, μBarVars[name].statevars, z)
    end
    μ_s = RJTVariables(μVars)
    return μ_s
end

"""
    expected_value(model::Model, diagram::InfluenceDiagram, μVars::RJTVariables)

Construct the RJT objective function.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `μVars::RJTVariables`: Vector of moments.

# Examples
```julia
EV = expected_value(model, diagram, μ_s)
```
"""
function expected_value(model::Model, diagram::InfluenceDiagram, μVars::RJTVariables)
    # Build the objective. The key observation here is that the information set
    # of a value node is always included in the previous cluster by construction.
    @expression(model, EV, 0)
    for V_name in keys(diagram.V)
        V_determining_node_name = diagram.RJT.arcs[findfirst(a -> a[2] == V_name, diagram.RJT.arcs)][1]
        V_determining_node_index_in_cluster = [findfirst(isequal(node), diagram.RJT.clusters[V_determining_node_name]) for node in diagram.I_j[V_name]]
        for index in CartesianIndices(μVars.data[V_determining_node_name].statevars)
            EV += diagram.Y[V_name][Tuple(index)[V_determining_node_index_in_cluster]...]*μVars.data[V_determining_node_name].statevars[index]
        end
    end
    return EV
end


"""
    conditional_value_at_risk(model::Model,
        diagram::InfluenceDiagram,
        μVars::RJTVariables,
        α::Float64)

Create a conditional value-at-risk (CVaR) objective based on RJT model.

The model can't have more than one value node.

# Arguments
- `model::Model`: JuMP model into which variables are added.
- `diagram::InfluenceDiagram`: Influence diagram structure.
- `μVars::RJTVariables`: Vector of moments.
- `α::Float64`: Probability level at which conditional value-at-risk is optimised.

# Examples
```julia
α = 0.05  # Parameter such that 0 ≤ α ≤ 1
CVaR = conditional_value_at_risk(model, diagram, μ_s, α)
```
"""
function conditional_value_at_risk(model::Model,
    diagram::InfluenceDiagram,
    μVars::RJTVariables,
    α::Float64)

    if length(diagram.V) != 1
        throw(DomainError("In order to create CVaR constraints, the number of value nodes should be 1."))
    end

    M = maximum(diagram.U.Y[1]) - minimum(diagram.U.Y[1])
    ϵ = minimum(diff(unique(diagram.U.Y[1]))) / 2

    η = @variable(model)
    ρ′_s = Dict{Float64, VariableRef}()

    value_node_name = first(n for (n, node) in diagram.Nodes if isa(node, ValueNode))
    #Assuming only one preceding node
    preceding_node_name = first(filter(tuple -> tuple[2]==collect(keys(diagram.V))[1], diagram.RJT.arcs))[1]

    #Finding the name and index of differing element between value nodes' information set and its preceding nodes rjt cluster. 
    #This is needed in conditional sums for constraints.
    missing_element = setdiff(diagram.RJT.clusters[preceding_node_name], diagram.Nodes[value_node_name].I_j)[1]
    index_to_remove = findfirst(x -> x == missing_element, diagram.RJT.clusters[preceding_node_name])

    statevars = μVars.data[preceding_node_name].statevars
    statevars_dims = collect(size(statevars))
    statevars_dims_ranges = [1:d for d in statevars_dims]
    
    function remove_index(old_tuple::NTuple{N, Int64}, index::Int64) where N
        return collect(ntuple(i -> i >= index ? old_tuple[i + 1] : old_tuple[i], N-1))
    end

    for u in unique(diagram.U.Y[1])
        λ = @variable(model, binary=true)
        λ′ = @variable(model, binary=true)
        ρ = @variable(model)
        ρ′ = @variable(model)
        @constraint(model, η - u ≤ M * λ)
        @constraint(model, η - u ≥ (M + ϵ) * λ - M)
        @constraint(model, η - u ≤ (M + ϵ) * λ′ - ϵ)
        @constraint(model, η - u ≥ M * (λ′ - 1))
        @constraint(model, 0 ≤ ρ)
        @constraint(model, 0 ≤ ρ′)
        @constraint(model, ρ ≤ λ)
        @constraint(model, ρ′ ≤ λ′)
        @constraint(model, ρ ≤ ρ′)

        p = @expression(model, sum(statevars[indices...] for indices in product(statevars_dims_ranges...) if diagram.U.Y[1][remove_index(indices, index_to_remove)...] == u))
        @constraint(model, ρ′ ≤ p)
        @constraint(model, (p - (1 - λ)) ≤ ρ)

        ρ′_s[u] = ρ′
    end

    @constraint(model, sum(values(ρ′_s)) == α)
    CVaR = @expression(model, (sum(ρ_bar * u for (u, ρ_bar) in ρ′_s)/α))

    return CVaR
end


"""
    generate_model(diagram::InfluenceDiagram; names::Bool=true, model_type::String, probability_cut::Bool=false)

Generate either decision programming based or RJT based variables and the respective objective function

# Examples
```julia-repl
julia> generate_model(diagram, model_type="RJT")
```
"""
function generate_model(
    diagram::InfluenceDiagram;
    names::Bool=true,
    model_type::String,
    forbidden_paths::Vector{ForbiddenPath}=ForbiddenPath[],
    fixed::FixedPath=Dict{Node, State}(),
    probability_cut::Bool=false
    )

    generate_diagram!(diagram)
    model = Model()
    z = DecisionVariables(model, diagram, names=names)
    if model_type=="RJT"
        variables = RJTVariables(model, diagram, z, names=names)
        EV = expected_value(model, diagram, variables)
        @objective(model, Max, EV)
    elseif model_type=="DP"
        variables = PathCompatibilityVariables(model, diagram, z, probability_cut = probability_cut, forbidden_paths = forbidden_paths, fixed=fixed)
        EV = expected_value(model, diagram, variables)
        @objective(model, Max, EV)     
    else
        error("Invalid model_type '$model_type'. It should be either 'RJT' or 'DP'.")
    end
    return model, z, variables
end
