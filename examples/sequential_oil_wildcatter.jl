using Revise
using DecisionProgramming
using JuMP, Distributions, Gurobi
diagram = InfluenceDiagram()
diagram.Cost = []

add_node!(diagram, ChanceNode("O", [], [ "wet", "dry"]))

add_node!(diagram, ChanceNode("R1", ["O"], ["wet", "dry"]))
add_node!(diagram, ChanceNode("R2", ["O"], ["wet", "dry"]))

add_node!(diagram, DecisionNode("D1", ["R1","R2"], ["yes", "no","another"],["R1","R2"]))
add_node!(diagram, DecisionNode("D2", ["R1","R2","D1"], ["yes","no"],["R1","R2"],[("R1",["D1"]),("R2",["D1"])]))

add_node!(diagram, ValueNode("V", ["O","D1","D2"]))

add_costs!(diagram,Costs(("R1","D1"), 0))
add_costs!(diagram,Costs(("R2","D1"), 0))

add_costs!(diagram,Costs(("R1","D2"), 25))
add_costs!(diagram,Costs(("R2","D2"), 25))


generate_arcs!(diagram)

add_probabilities!(diagram, "O", [0.01,0.99])
X_R = ProbabilityMatrix(diagram, "R1")
X_R["wet","wet"] = 0.6
X_R["dry","wet"] =  0.07
X_R["dry","dry"] = 0.93
X_R["wet","dry"] = 0.4
add_probabilities!(diagram, "R1", X_R)

X_R = ProbabilityMatrix(diagram, "R2")
X_R["wet","wet"] = 0.93
X_R["dry","wet"] =  0.35
X_R["dry","dry"] = 0.65
X_R["wet","dry"] = 0.07
add_probabilities!(diagram, "R2", X_R)



Y_V = UtilityMatrix(diagram, "V")
Y_V["wet", "yes",:] = [20000,0]
Y_V["wet", "no",:] = [0,0]
Y_V["wet", "another",:] = [19900,-100]

Y_V["dry", "yes",:] = [-1000,0]
Y_V["dry", "no",:] = [0,0]
Y_V["dry", "another",:] = [-1100,-100]
add_utilities!(diagram, "V", Y_V)

generate_diagram!(diagram,positive_path_utility = true)

model = Model()
z = DecisionVariables(model, diagram, augmented_states = true,names=true, name = "z")
x_s = PathCompatibilityVariables(model, diagram, z, names=true, name = "s", augmented_states = true, probability_cut = false)
(x_x,x_xx) = StateDependentAugmentedStateVariables(model,diagram,z,x_s,names=true,name="x")
EV = expected_value(model, diagram, x_s,x_x = x_x, x_xx = x_xx)
@constraint(model,x_x[(2,4)] <= x_xx[(2,5)][(1,)])
@constraint(model,x_x[(2,4)] <= x_xx[(2,5)][(2,)])
@constraint(model,x_x[(2,4)] <= x_xx[(2,5)][(3,)])
@constraint(model,x_x[(3,4)] <= x_xx[(3,5)][(1,)])
@constraint(model,x_x[(3,4)] <= x_xx[(3,5)][(2,)])
@constraint(model,x_x[(3,4)] <= x_xx[(3,5)][(3,)])
@constraint(model,x_x[(2,4)] + x_x[(3,4)] <= 1)
@constraint(model,x_xx[(3,5)][(3,)] + x_xx[(2,5)][(3,)] <= 1 + x_x[(2,4)] + x_x[(3,4)])
@constraint(model,x_xx[(3,5)][(2,)] + x_xx[(2,5)][(2,)] <= 1 + x_x[(2,4)] + x_x[(3,4)])
@constraint(model,x_xx[(3,5)][(1,)] + x_xx[(2,5)][(1,)] <= 1 + x_x[(2,4)] + x_x[(3,4)])
for (d, z_d) in zip(z.D, z.z)
    if d == 5
        @constraint(model, z_d[10] == 0)
        @constraint(model, z_d[11] == 0)
        @constraint(model, z_d[12] == 0)
        @constraint(model, z_d[13] == 0)
        @constraint(model, z_d[14] == 0)
        @constraint(model, z_d[15] == 0)
        @constraint(model, z_d[16] == 0)
        @constraint(model, z_d[17] == 0)
        @constraint(model, z_d[18] == 0)
        @constraint(model, z_d[28] == 0)
        @constraint(model, z_d[29] == 0)
        @constraint(model, z_d[30] == 0)
        @constraint(model, z_d[31] == 0)
        @constraint(model, z_d[32] == 0)
        @constraint(model, z_d[33] == 0)
        @constraint(model, z_d[34] == 0)
        @constraint(model, z_d[35] == 0)
        @constraint(model, z_d[36] == 0)
    end
end

@objective(model, Max, EV)
optimizer = optimizer_with_attributes(
        () -> Gurobi.Optimizer(Gurobi.Env()),
            "IntFeasTol"      => 1e-9,
        )
set_optimizer(model, optimizer)
optimize!(model)
Z = DecisionStrategy(z)
U_distribution = AugmentedUtilityDistribution(diagram, Z,x_s,x_x = x_x,x_xx = x_xx)
S_probabilities = StateProbabilities(diagram, Z,x_s)
print_state_probabilities(diagram,S_probabilities,["O","R1","R2","D1","D2"])
print_decision_strategy(diagram, Z, S_probabilities, augmented_states = true)
print_utility_distribution(U_distribution)
print_statistics(U_distribution)
print_information_structure(diagram,x_x,x_xx = x_xx)