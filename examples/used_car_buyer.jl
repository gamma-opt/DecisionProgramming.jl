
using Logging
using JuMP, Gurobi
using DecisionProgramming

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

add_node!(diagram, ChanceNode("O", [], ["lemon", "peach"]))
add_node!(diagram, ChanceNode("R", ["O", "T"], ["no test", "lemon", "peach"]))

add_node!(diagram, DecisionNode("T", [], ["no test", "test"]))
add_node!(diagram, DecisionNode("A", ["R"], ["buy without guarantee", "buy with guarantee", "don't buy"]))

add_node!(diagram, ValueNode("V1", ["T"]))
add_node!(diagram, ValueNode("V2", ["A"]))
add_node!(diagram, ValueNode("V3", ["O", "A"]))

generate_arcs!(diagram)

X_O = ProbabilityMatrix(diagram, "O")
set_probability!(X_O, ["peach"], 0.8)
set_probability!(X_O, ["lemon"], 0.2)
add_probabilities!(diagram, "O", X_O)


X_R = ProbabilityMatrix(diagram, "R")
X_R[1, 1, :] = [1,0,0]
X_R[1, 2, :] = [0,1,0]
X_R[2, 1, :] = [1,0,0]
X_R[2, 2, :] = [0,0,1]
add_probabilities!(diagram, "R", X_R)

add_utilities!(diagram, "V1", [0, -25])
add_utilities!(diagram, "V2", [100, 40, 0])

Y_V3 = UtilityMatrix(diagram, "V3")
set_utility!(Y_V3, ["lemon", "buy without guarantee"], -200)
set_utility!(Y_V3, ["lemon", "buy with guarantee"], 0)
set_utility!(Y_V3, ["lemon", "don't buy"], 0)
set_utility!(Y_V3, ["peach", :], [-40, -20, 0])
add_utilities!(diagram, "V3", Y_V3)

generate_diagram!(diagram)


@info("Creating the decision model.")
model = Model()
z = DecisionVariables(model, diagram)
x_s = PathCompatibilityVariables(model, diagram, z)
EV = expected_value(model, diagram, x_s)
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
U_distribution = UtilityDistribution(diagram, Z)

@info("Printing decision strategy:")
print_decision_strategy(diagram, Z, S_probabilities)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

@info("Printing expected utility.")
print_statistics(U_distribution)
