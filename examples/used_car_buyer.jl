using Logging
using JuMP, HiGHS
using DecisionProgramming

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

add_node!(diagram, ChanceNode("O", [], ["lemon", "peach"]))
add_node!(diagram, DecisionNode("T", [], ["no test", "test"]))
add_node!(diagram, ChanceNode("R", ["O", "T"], ["no test", "lemon", "peach"]))
add_node!(diagram, DecisionNode("A", ["R"], ["buy without guarantee", "buy with guarantee", "don't buy"]))

add_node!(diagram, ValueNode("V1", ["T"]))
add_node!(diagram, ValueNode("V2", ["A"]))
add_node!(diagram, ValueNode("V3", ["O", "A"]))

generate_arcs!(diagram)

X_O = ProbabilityMatrix(diagram, "O")
X_O["peach"] = 0.8
X_O["lemon"] = 0.2
add_probabilities!(diagram, "O", X_O)

X_R = ProbabilityMatrix(diagram, "R")
X_R["lemon", "no test", :] = [1,0,0]
X_R["lemon", "test", :] = [0,1,0]
X_R["peach", "no test", :] = [1,0,0]
X_R["peach", "test", :] = [0,0,1]
add_probabilities!(diagram, "R", X_R)

Y_V1 = UtilityMatrix(diagram, "V1")
Y_V1["test"] = -25
Y_V1["no test"] = 0
add_utilities!(diagram, "V1", Y_V1)

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

generate_diagram!(diagram)


@info("Creating the decision model.")
model = Model()
z = DecisionVariables(model, diagram)
println("z:")
println(z)

"""
x_s = PathCompatibilityVariables(model, diagram, z)
EV = expected_value(model, diagram, x_s)
@objective(model, Max, EV)
"""

μVars, μ_barVars = cluster_variables_and_constraints(model, diagram, z)
RJT_objective(model, diagram, μVars)
println("model:")
println(model)


@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> HiGHS.Optimizer()
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(diagram, z)
S_probabilities = StateProbabilities(diagram, Z)
U_distribution = UtilityDistribution(diagram, Z)

@info("Printing decision strategy:")
print_decision_strategy(diagram, Z, S_probabilities)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

@info("Printing expected utility.")
print_statistics(U_distribution)
