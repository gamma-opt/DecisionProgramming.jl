using Logging
using JuMP, HiGHS
#using DecisionProgramming



const N = 4

diagram = InfluenceDiagram()

add_node!(diagram, ChanceNode("H1", [], ["ill", "healthy"]))
for i in 1:N-1
    # Testing result
    add_node!(diagram, ChanceNode("T$i", ["H$i"], ["positive", "negative"]))
    # Decision to treat
    add_node!(diagram, DecisionNode("D$i", ["T$i"], ["treat", "pass"]))
    # Cost of treatment
    add_node!(diagram, ValueNode("C$i", ["D$i"]))
    # Health of next period
    add_node!(diagram, ChanceNode("H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"]))
end
add_node!(diagram, ValueNode("MP", ["H$N"]))

generate_arcs!(diagram);



# Add probabilities for node H1
add_probabilities!(diagram, "H1", [0.1, 0.9])

# Declare proability matrix for health nodes H_2, ... H_N-1, which have identical information sets and states
X_H = ProbabilityMatrix(diagram, "H2")
X_H["healthy", "pass", :] = [0.2, 0.8]
X_H["healthy", "treat", :] = [0.1, 0.9]
X_H["ill", "pass", :] = [0.9, 0.1]
X_H["ill", "treat", :] = [0.5, 0.5]

# Declare proability matrix for test result nodes T_1...T_N
X_T = ProbabilityMatrix(diagram, "T1")
X_T["ill", "positive"] = 0.8
X_T["ill", "negative"] = 0.2
X_T["healthy", "negative"] = 0.9
X_T["healthy", "positive"] = 0.1

for i in 1:N-1
    add_probabilities!(diagram, "T$i", X_T)
    add_probabilities!(diagram, "H$(i+1)", X_H)
end

for i in 1:N-1
    add_utilities!(diagram, "C$i", [-100.0, 0.0])
end

add_utilities!(diagram, "MP", [300.0, 1000.0])

generate_diagram!(diagram);




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
    return C_rjt, A_rjt
end




# Using the influence diagram and decision variables z from DecisionProgramming.jl,  
# adds the variables and constraints of the corresponding RJT model
function cluster_variables_and_constraints(model, diagram, z)

    # I would prefer that we made some minor modifications to the diagram structure, 
    # having these as dictionaries makes things a lot easier in the model formulation
    S = Dict{String, Vector{String}}()
    idx = 1
    for node in diagram.Nodes
        if !isa(node, ValueNode)
            S[node.name] = node.states
            idx+=1
        end
    end    

    I_j = Dict{String, Vector{String}}()
    for (idx, name) in enumerate(diagram.Names)
        I_j[name] = diagram.Names[diagram.I_j[idx]]
    end

    Nodes = Dict{String, AbstractNode}()
    for node in diagram.Nodes
        Nodes[node.name] = node
    end

    States = Dict{String, Vector{String}}()
    for (idx, name) in enumerate(diagram.Names)
        if !isa(Nodes[name], ValueNode)
            States[name] = diagram.States[idx]
        else
            States[name] = []
        end
    end

    X = Dict{String, Probabilities}()
    idx = 1
    for name in diagram.Names
        if isa(Nodes[name], ChanceNode)
            X[name] = diagram.X[idx]
            idx += 1
        end
    end
    
    Y = Dict{String, Utilities}()
    idx = 1
    for name in diagram.Names
        if isa(Nodes[name], ValueNode)
            Y[name] = diagram.Y[idx]
            idx += 1
        end
    end

    z_dict = Dict{String, Array{VariableRef}}()
    idx = 1
    for name in diagram.Names
        if isa(Nodes[name], DecisionNode)
            z_dict[name] = z.z[idx]
            idx += 1
        end
    end


    # Get the RJT structure
    C_rjt, A_rjt = ID_to_RJT(diagram)

    # Variables corresponding to the nodes in the RJT
    μ = Dict{String, Array{VariableRef}}()
    for j in keys(C_rjt)
        if !isa(Nodes[j], ValueNode)
            μ[j] = Array{VariableRef}(undef, Tuple(length.([getindex.(Ref(S), C_rjt[j])]...)))
            for index in CartesianIndices(μ[j])
                μ[j][index] = @variable(model, base_name="μ_$j($(join(Tuple(index),',')))", lower_bound=0)
            end
            # Probability distributions μ sum to 1
            @constraint(model, sum(μ[j]) == 1)
        end
    end

    for a in A_rjt
        if !isa(Nodes[a[2]], ValueNode)
            intersection = C_rjt[a[1]] ∩ C_rjt[a[2]]
            C1_minus_C2 = Tuple(setdiff(collect(1:length(C_rjt[a[1]])), indexin(intersection, C_rjt[a[1]])))
            C2_minus_C1 = Tuple(setdiff(collect(1:length(C_rjt[a[2]])), indexin(intersection, C_rjt[a[2]])))
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
            for index in CartesianIndices(μ_breve[j])
                # Moments μ_{\breve{C}_v} (the moments from above, but with the last variable dropped out)
                μ_breve[j][index] = @variable(model, base_name="μ_breve_$j($(join(Tuple(index),',')))", lower_bound=0)
                # μ_{\breve{C}_v} = ∑_{x_v} μ_{C_v}
                @constraint(model, μ_breve[j][index] .== dropdims(sum(μ[j], dims=findfirst(isequal(j), C_rjt[j])), dims=findfirst(isequal(j), C_rjt[j]))[index])
            end
        end
    end

    # Add in the conditional probabilities and decision strategies
    for name in diagram.Names 
        if !isa(Nodes[name], ValueNode) # In our structure, value nodes are not stochastic and the whole objective thing doesn't really work in this context
            I_j_mapping = [findfirst(isequal(node), C_rjt[name]) for node in I_j[name]] # Map the information set to the variables in the cluster
            for index in CartesianIndices(μ_breve[name])
                for s_j in 1:length(States[name])
                    if isa(Nodes[name], ChanceNode)
                        # μ_{C_v} = p*μ_{\breve{C}_v}
                        @constraint(model, μ[name][Tuple(index)...,s_j] == X[name][Tuple(index)[I_j_mapping]...,s_j]*μ_breve[name][index])
                    elseif isa(Nodes[name], DecisionNode)
                        # μ_{C_v} = z*μ_{\breve{C}_v}
                        println("nakki")
                        println(z_dict[name][Tuple(index)[I_j_mapping]...,s_j]*μ_breve[name][index])
                        @constraint(model, μ[name][Tuple(index)...,s_j] == z_dict[name][Tuple(index)[I_j_mapping]...,s_j]*μ_breve[name][index])
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
            I_j_mapping = [findfirst(isequal(node), C_rjt[i]) for node in I_j[j]]
            for index in CartesianIndices(μ[i])
                set_objective_coefficient(model, μ[i][index], Y[j][Tuple(index)[I_j_mapping]...])
            end
        end
    end
    
    return μ
end




model = Model()
# set_silent(model)
optimizer = optimizer_with_attributes(
    () -> HiGHS.Optimizer()
    # "DualReductions"  => 0,
)
set_optimizer(model, optimizer)

z = DecisionVariables(model, diagram)

μ = cluster_variables_and_constraints(model, diagram, z)
println("nakki")
model


optimize!(model)


Z = DecisionStrategy(z)
# S_probabilities = StateProbabilities(diagram, Z)
U_distribution = UtilityDistribution(diagram, Z);


# print_decision_strategy(diagram, Z, S_probabilities)

print_utility_distribution(U_distribution)

print_statistics(U_distribution)