using DecisionProgramming
using JuMP, Gurobi
diagram = InfluenceDiagram()
diagram.Cost = []

add_node!(diagram, ChanceNode("C", [], [ "no cancer", "early stage","advanced stage"]))

add_node!(diagram, ChanceNode("S", ["C"], ["no cancer", "cancer"]))
add_node!(diagram, ChanceNode("T", ["C"], ["no cancer", "early stage","advanced stage"]))

add_node!(diagram, DecisionNode("D1", ["S","T"], ["no treatment","chemo","surgery and chemo"],["T"],[("T",["S"])]))

add_node!(diagram, ChanceNode("H", ["C","D1"], ["healthy", "ill"]))


add_node!(diagram, ValueNode("V", ["H"]))

add_costs!(diagram,Costs(("T","D1"), 5))


generate_arcs!(diagram)

add_probabilities!(diagram, "C", [0.7,0.2,0.1])

X_R = ProbabilityMatrix(diagram, "T")
X_R["no cancer","no cancer"] = 0.95
X_R["no cancer","early stage"] =  0.03
X_R["no cancer","advanced stage"] =  0.02
X_R["early stage","no cancer"] = 0.03
X_R["early stage","early stage"] =  0.95
X_R["early stage","advanced stage"] =  0.02
X_R["advanced stage","no cancer"] = 0.01
X_R["advanced stage","early stage"] =  0.04
X_R["advanced stage","advanced stage"] =  0.95
add_probabilities!(diagram, "T", X_R)

X_R2 = ProbabilityMatrix(diagram, "S")
X_R2["no cancer","no cancer"] = 0.99
X_R2["no cancer","cancer"] =  0.01
X_R2["early stage","no cancer"] = 0.01
X_R2["early stage","cancer"] =  0.99
X_R2["advanced stage","no cancer"] = 0.01
X_R2["advanced stage","cancer"] =  0.99
add_probabilities!(diagram, "S", X_R2)



X_R3 = ProbabilityMatrix(diagram, "H")
X_R3["no cancer","no treatment",:] = [0.99, 0.01]
X_R3["no cancer","chemo",:] = [0.8, 0.2]
X_R3["no cancer","surgery and chemo",:] = [0.5, 0.5]
X_R3["early stage","no treatment",:] = [0.1, 0.9]
X_R3["early stage","chemo",:] = [0.8, 0.2]
X_R3["early stage","surgery and chemo",:] = [0.4, 0.6]
X_R3["advanced stage","no treatment",:] = [0, 1]
X_R3["advanced stage","chemo",:] = [0.2, 0.8]
X_R3["advanced stage","surgery and chemo",:] = [0.5, 0.5]
add_probabilities!(diagram, "H", X_R3)

add_utilities!(diagram, "V", [20000,0])

generate_diagram!(diagram,positive_path_utility = true)

model = Model()
z = DecisionVariables(model, diagram, augmented_states = true,names=true, name = "z")
x_s = PathCompatibilityVariables(model, diagram, z, names=true, name = "s", augmented_states = true, probability_cut = false)
(x_x,x_xx) = StateDependentAugmentedStateVariables(model,diagram,z,x_s,names=true,name="x")
EV = expected_value(model, diagram, x_s,x_x = x_x, x_xx = x_xx)

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
print_decision_strategy(diagram, Z, S_probabilities, augmented_states = true)
print_utility_distribution(U_distribution)
print_statistics(U_distribution)
print_information_structure(diagram,x_x,x_xx = x_xx)