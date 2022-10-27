using Logging
using JuMP, Gurobi
using DecisionProgramming

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()
diagram.Cost = []

add_node!(diagram, ChanceNode("O", [], ["lemon", "peach"]))
add_node!(diagram, ChanceNode("R", ["O"], ["lemon", "peach"]))
add_node!(diagram, DecisionNode("A", ["R"], ["buy without guarantee", "buy with guarantee", "don't buy"],["R"]))

add_node!(diagram, ValueNode("V2", ["A"]))
add_node!(diagram, ValueNode("V3", ["O", "A"]))

add_costs!(diagram,Costs(("R","A"), 25))

generate_arcs!(diagram)

X_O = ProbabilityMatrix(diagram, "O")
X_O["peach"] = 0.8
X_O["lemon"] = 0.2
add_probabilities!(diagram, "O", X_O)

X_R = ProbabilityMatrix(diagram, "R")
X_R["lemon", :] = [1,0]
X_R["peach", :] = [0,1]
add_probabilities!(diagram, "R", X_R)

Y_V2 = UtilityMatrix(diagram, "V2")
Y_V2["buy without guarantee"] = 100
Y_V2["buy with guarantee"] = 40
Y_V2["don't buy"] = 0
add_utilities!(diagram, "V2", Y_V2)

Y_V3 = UtilityMatrix(diagram, "V3")
Y_V3["lemon", "buy without guarantee"] = -200
Y_V3["lemon", "buy with guarantee"] = 0
Y_V3["lemon", "don't buy"] = 0
Y_V3["peach", :] = [-40, -20, 0]
add_utilities!(diagram, "V3", Y_V3)

generate_diagram!(diagram, positive_path_utility = true)


@info("Creating the decision model.")
model = Model()
z = DecisionVariables(model, diagram, names=true, name = "z")
x_s = PathCompatibilityVariables(model, diagram, z, names=true, name = "s", probability_cut = false)
x_x = InformationConstraintVariables(model,diagram,z,x_s,names=true,name="x")
EV = expected_value(model, diagram, x_s,x_x = x_x)
@objective(model, Max, EV)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(z)
S_probabilities = StateProbabilities(diagram, Z)
U_distribution = UtilityDistribution(diagram, Z,x_x = x_x)

@info("Printing decision strategy:")
print_decision_strategy(diagram, Z, S_probabilities)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

@info("Printing expected utility.")
print_statistics(U_distribution)

@info("Printing information structure")
print_information_structure(diagram,x_x)
