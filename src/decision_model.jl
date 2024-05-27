using JuMP

function decision_variable(model::Model, S::States, d::Node, I_d::Vector{Node}, base_name::String="")
    # Create decision variables.
    println(d)
    println(I_d)
    #println("base_name" * base_name)
    dims = S[[I_d; d]]
    println(dims)
    z_d = Array{VariableRef}(undef, dims...)
    println(z_d)
    for s in paths(dims)
        z_d[s...] = @variable(model, binary=true, base_name=base_name)
    end
    println(z_d)
    # Constraints to one decision per decision strategy.
    for s_I in paths(S[I_d])
        println(s_I)
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
    println(diagram.D)
    println(diagram.I_j[diagram.D])
    println(diagram.S)
    println(diagram.States)
    #println(diagram.d)
    #println(I_d)
    println(zip(diagram.D, diagram.I_j[diagram.D]))
    DecisionVariables(diagram.D, diagram.I_j[diagram.D], [decision_variable(model, diagram.S, d, I_d, (names ? "$(name)_$(d.j)$(s)" : "")) for (d, I_d) in zip(diagram.D, diagram.I_j[diagram.D])])
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
        #println(s_d_s_Id...)
        #println(z[s_d_s_Id...])
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
    println("N: $N")
    variables_x_s = Dict{Path{N}, VariableRef}(
        s => path_compatibility_variable(model, (names ? "$(name)$(s)" : ""))
        for s in paths(diagram.S, fixed)
        if !iszero(diagram.P(s)) && !is_forbidden(s, forbidden_paths)
    )

    x_s = PathCompatibilityVariables{N}(variables_x_s)
    println("x_s: $x_s")

    # Add decision strategy constraints for each decision node
    #println("nakki")
    #println(z)
    #println(z_d)
    #println(z.D)
    #println(z.z)
    for (d, z_d) in zip(z.D, z.z)
        #println(z_d)
        decision_strategy_constraint(model, diagram.S, d, diagram.I_j[d], z.D, z_d, x_s)
    end

    if probability_cut
        @constraint(model, sum(x * diagram.P(s) * probability_scale_factor for (s, x) in x_s) == 1.0 * probability_scale_factor)
    end

    x_s
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

    @expression(model, sum(diagram.P(s) * x * diagram.U(s, diagram.translation) for (s, x) in x_s))
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


# RJT MODEL

# Implements Algorithm 1 from Parmentier et al. (2020)
function ID_to_RJT(diagram)
    C_rjt = Dict{String, Vector{String}}()
    A_rjt = []
    namelist = [node.name for node in diagram.Nodes]
    for j in length(diagram.Nodes):-1:1
        C_j = copy(diagram.Nodes[j].I_j)
        push!(C_j, namelist[j])
        for a in A_rjt 
            if a[1] == namelist[j]
                push!(C_j, setdiff(C_rjt[a[2]], [a[2]])...)
            end
        end
        C_j = unique(C_j)
        C_j_aux = sort([(elem, findfirst(isequal(elem), namelist)) for elem in C_j], by = last)
        C_j = [C_j_tuple[1] for C_j_tuple in C_j_aux]
        C_rjt[namelist[j]] = C_j
        
        if length(C_rjt[namelist[j]]) > 1
            u = maximum([findfirst(isequal(name), namelist) for name in setdiff(C_j, [namelist[j]])])
            push!(A_rjt, (namelist[u], namelist[j]))
        end
    end
    println(C_rjt)
    println(A_rjt)
    println("")
    return C_rjt, A_rjt
end






# Using the influence diagram and decision variables z from DecisionProgramming.jl,  
# adds the variables and constraints of the corresponding RJT model
function cluster_variables_and_constraints(model, diagram, z)

    # I would prefer that we made some minor modifications to the diagram structure, 
    # having these as dictionaries makes things a lot easier in the model formulation
    # Muiden kuin value nodejen statet dictionaryyn
    S = Dict{String, Vector{String}}()
    idx = 1
    for node in diagram.Nodes
        if !isa(node, ValueNode)
            S[node.name] = node.states
            idx+=1
        end
    end    
    println(diagram.Nodes)
    println("S: $S")
    println("")

    I_j = Dict{String, Vector{String}}()
    for (idx, name) in enumerate(diagram.Names)
        I_j[name] = diagram.Names[diagram.I_j[idx]]
    end
    println(I_j)
    println("")

    Nodes = Dict{String, AbstractNode}()
    for node in diagram.Nodes
        Nodes[node.name] = node
    end
    println(Nodes)
    println("")

    States = Dict{String, Vector{String}}()
    for (idx, name) in enumerate(diagram.Names)
        if !isa(Nodes[name], ValueNode)
            States[name] = diagram.States[idx]
        else
            States[name] = []
        end
    end
    println(States)
    println("")

    X = Dict{String, Probabilities}()
    idx = 1
    for name in diagram.Names
        if isa(Nodes[name], ChanceNode)
            X[name] = diagram.X[idx]
            idx += 1
        end
    end
    println(X)
    println("")
    
    Y = Dict{String, Utilities}()
    idx = 1
    for name in diagram.Names
        if isa(Nodes[name], ValueNode)
            Y[name] = diagram.Y[idx]
            idx += 1
        end
    end
    println(Y)
    println("")

    z_dict = Dict{String, Array{VariableRef}}()
    idx = 1
    for name in diagram.Names
        if isa(Nodes[name], DecisionNode)
            z_dict[name] = z.z[idx]
            idx += 1
        end
    end
    println("z_dict: $z_dict")
    println("")

    # Get the RJT structure
    C_rjt, A_rjt = ID_to_RJT(diagram)

    # Variables corresponding to the nodes in the RJT
    μ = Dict{String, Array{VariableRef}}()
    for j in keys(C_rjt)
        if !isa(Nodes[j], ValueNode)
            μ[j] = Array{VariableRef}(undef, Tuple(length.([getindex.(Ref(S), C_rjt[j])]...)))
            for index in CartesianIndices(μ[j])
                μ[j][index] = @variable(model, base_name="μ_$j($(join(Tuple(index),',')))", lower_bound=0)
                #println(μ[j][index])
            end
            # Probability distributions μ sum to 1
            @constraint(model, sum(μ[j]) == 1)
        end
    end

    println(μ)
    println("")

    for a in A_rjt
        if !isa(Nodes[a[2]], ValueNode)
            intersection = C_rjt[a[1]] ∩ C_rjt[a[2]]
            println(intersection)
            C1_minus_C2 = Tuple(setdiff(collect(1:length(C_rjt[a[1]])), indexin(intersection, C_rjt[a[1]])))
            println(C1_minus_C2)
            C2_minus_C1 = Tuple(setdiff(collect(1:length(C_rjt[a[2]])), indexin(intersection, C_rjt[a[2]])))
            println(C2_minus_C1)
            println("")
            @constraint(model, 
                dropdims(sum(μ[a[1]], dims=C1_minus_C2), dims=C1_minus_C2) .== 
                dropdims(sum(μ[a[2]], dims=C2_minus_C1), dims=C2_minus_C1))
        end
    end

    # Variables μ_{\breve{C}_v} = ∑_{x_v} μ_{C_v}
    μ_breve = Dict{String, Array{VariableRef}}()
    for j in keys(C_rjt)
        if !isa(Nodes[j], ValueNode)
            μ_breve[j] = Array{VariableRef}(undef, Tuple(length.([getindex.(Ref(S), setdiff(C_rjt[j], [j]))]...)))
            println(μ_breve[j])
            for index in CartesianIndices(μ_breve[j])
                # Moments μ_{\breve{C}_v} (the moments from above, but with the last variable dropped out)
                μ_breve[j][index] = @variable(model, base_name="μ_breve_$j($(join(Tuple(index),',')))", lower_bound=0)
                # μ_{\breve{C}_v} = ∑_{x_v} μ_{C_v}
                @constraint(model, μ_breve[j][index] .== dropdims(sum(μ[j], dims=findfirst(isequal(j), C_rjt[j])), dims=findfirst(isequal(j), C_rjt[j]))[index])
            end
        end
    end

    println(μ_breve)
    println("")
    println(diagram.Names)
    println("")
    println(I_j)

    # Add in the conditional probabilities and decision strategies
    for name in diagram.Names 
        if !isa(Nodes[name], ValueNode) # In our structure, value nodes are not stochastic and the whole objective thing doesn't really work in this context
            I_j_mapping = [findfirst(isequal(node), C_rjt[name]) for node in I_j[name]] # Map the information set to the variables in the cluster
            #println(I_j_mapping)
            for index in CartesianIndices(μ_breve[name])
                println(μ_breve[name])
                for s_j in 1:length(States[name])
                    if isa(Nodes[name], ChanceNode)
                        # μ_{C_v} = p*μ_{\breve{C}_v}
                        @constraint(model, μ[name][Tuple(index)...,s_j] == X[name][Tuple(index)[I_j_mapping]...,s_j]*μ_breve[name][index])
                        #println(μ[name][Tuple(index)...,s_j])
                        #println(X[name][Tuple(index)[I_j_mapping]...,s_j])
                        #println(μ_breve[name][index])
                    elseif isa(Nodes[name], DecisionNode)
                        # μ_{C_v} ≤ z
                        @constraint(model, μ[name][Tuple(index)...,s_j] <= z_dict[name][Tuple(index)[I_j_mapping]...,s_j])
                        println(μ[name][Tuple(index)...,s_j])
                        println(z_dict[name][Tuple(index)[I_j_mapping]...,s_j])
                    end
                end
            end
        end
    end

    # Build the objective. The key observation here is that the information set
    # of a value node is always included in the previous cluster by construction
    @objective(model, Max, 0)
    for j in keys(C_rjt)
        if isa(Nodes[j], ValueNode)
            i = A_rjt[findfirst(a -> a[2] == j, A_rjt)][1]
            println("nakki")
            println(i)
            I_j_mapping = [findfirst(isequal(node), C_rjt[i]) for node in I_j[j]]
            println(I_j_mapping)
            for index in CartesianIndices(μ[i])
                set_objective_coefficient(model, μ[i][index], Y[j][Tuple(index)[I_j_mapping]...])
            end
        end
    end
    
    return μ
end