using Pkg
using DecisionProgramming
using JuMP
using Distributions
using PlotlyJS
using Gurobi

diagram = InfluenceDiagram()
diagram.Cost = []

add_node!(diagram, ChanceNode("H1", [], ["ill", "healthy"]))
add_node!(diagram, ChanceNode("TA1", ["H1"], ["positive", "negative"]))
add_node!(diagram, ChanceNode("TB1", ["H1"], ["positive", "negative"]))
add_node!(diagram, DecisionNode("D1", ["TA1","TB1"], ["treat", "pass"],["TA1","TB1"] ))
add_node!(diagram, ValueNode("C1", ["D1"]))
add_node!(diagram, ChanceNode("H2", ["H1", "D1"], ["ill", "healthy"]))
add_node!(diagram, ChanceNode("TA2", ["H2"], ["positive", "negative"]))
add_node!(diagram, ChanceNode("TB2", ["H2"], ["positive", "negative"]))
add_node!(diagram, DecisionNode("D2", ["TA2","TB2"], ["treat", "pass"],["TA2","TB2"]))
add_node!(diagram, ChanceNode("H3", ["H2", "D2"], ["ill", "healthy"]))
add_node!(diagram, ChanceNode("TA3", ["H3"], ["positive", "negative"]))
add_node!(diagram, ChanceNode("TB3", ["H3"], ["positive", "negative"]))
add_node!(diagram, DecisionNode("D3", ["TA3","TB3"], ["treat", "pass"],["TA3","TB3"]))
add_node!(diagram, ValueNode("C2", ["D2"]))
add_node!(diagram, ValueNode("C3", ["D3"]))
add_node!(diagram, ChanceNode("H4", ["H3", "D3"], ["ill", "healthy"]))
add_node!(diagram, ValueNode("MP", ["H4"]))

add_costs!(diagram,Costs(("TA1","D1"), 3))
add_costs!(diagram,Costs(("TA2","D2"), 3))
add_costs!(diagram,Costs(("TA3","D3"), 3))
add_costs!(diagram,Costs(("TB1","D1"), 8))
add_costs!(diagram,Costs(("TB2","D2"), 8))
add_costs!(diagram,Costs(("TB3","D3"), 8))

generate_arcs!(diagram)

# Add probabilities for node H1
add_probabilities!(diagram, "H1", [0.1, 0.9])

# Declare proability matrix for health nodes H_2, ... H_N-1, which have identical information sets and states
X_H = ProbabilityMatrix(diagram, "H2")
X_H["healthy", "pass", :] = [0.2, 0.8]
X_H["healthy", "treat", :] = [0.1, 0.9]
X_H["ill", "pass", :] = [0.9, 0.1]
X_H["ill", "treat", :] = [0.5, 0.5]

add_probabilities!(diagram, "H2", X_H)
add_probabilities!(diagram, "H3", X_H)
add_probabilities!(diagram, "H4", X_H)

# Declare proability matrix for test result nodes T_1...T_N

X_T3 = ProbabilityMatrix(diagram, "TA1")
X_T3["ill", "positive"] = 0.8
X_T3["ill", "negative"] = 0.2
X_T3["healthy", "positive"] = 0.1
X_T3["healthy", "negative"] = 0.9
add_probabilities!(diagram, "TA1", X_T3)
add_probabilities!(diagram, "TA2", X_T3)
add_probabilities!(diagram, "TA3", X_T3)

X_T2 = ProbabilityMatrix(diagram, "TB2")
X_T2["ill", "positive"] = 0.95
X_T2["ill", "negative"] = 0.05
X_T2["healthy", "positive"] = 0.05
X_T2["healthy", "negative"] = 0.95
add_probabilities!(diagram, "TB2", X_T2)
add_probabilities!(diagram, "TB1", X_T2)
add_probabilities!(diagram, "TB3", X_T2)



add_utilities!(diagram, "C1", [-100.0, 0.0])
add_utilities!(diagram, "C2", [-100.0, 0.0])
add_utilities!(diagram, "C3", [-100.0, 0.0])


add_utilities!(diagram, "MP", [300.0, 1000.0])

generate_diagram!(diagram,positive_path_utility = false)


model3 = Model()
z = DecisionVariables(model3, diagram, augmented_states = true, names=true, name = "z")
x_s = PathCompatibilityVariables(model3, diagram, z, augmented_states = true, names=true, name = "s", probability_cut = false)
x_x = AugmentedStateVariables(model3,diagram,z,x_s,names=true,name="x")
EV = expected_value(model3, diagram, x_s,x_x = x_x)
@objective(model3, Max, EV)

optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
        "IntFeasTol"      => 1e-9,
    )
set_optimizer(model3, optimizer)
set_silent(model3)
optimize!(model3)


Z = DecisionStrategy(z)
U_distribution = AugmentedUtilityDistribution(diagram, Z,x_s,x_x = x_x)
S_probabilities = StateProbabilities(diagram, Z,x_s)
print_decision_strategy(diagram, Z, S_probabilities,augmented_states = true)
print_utility_distribution(U_distribution)
print_statistics(U_distribution)
print_information_structure(diagram,x_x)