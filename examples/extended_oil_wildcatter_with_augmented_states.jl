using Logging
using JuMP, GLPK
using DecisionProgramming

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

diagram.Cost = []

add_node!(diagram, ChanceNode("O", [], [ "yes", "no"]))

add_node!(diagram, ChanceNode("R1", ["O"], ["yes", "no"]))
add_node!(diagram, ChanceNode("R2", ["O"], ["yes", "no"]))
add_node!(diagram, ChanceNode("R3", ["O"], ["yes", "no"]))

add_node!(diagram, DecisionNode("D", ["R1","R2","R3"], ["yes", "no"],["R1","R2","R3"]))

add_node!(diagram, ValueNode("V",["O","D"]))

add_costs!(diagram,Costs(("R1","D"), 1))
add_costs!(diagram,Costs(("R2","D"), 1))
add_costs!(diagram,Costs(("R3","D"), 1))

generate_arcs!(diagram)

# Add probabilities for node H1
add_probabilities!(diagram, "O", [0.1, 0.9])

# Declare proability matrix for reports
X_R1 = ProbabilityMatrix(diagram, "R1")
X_R1["yes", "yes"] = 0.85
X_R1["yes", "no"] = 0.15
X_R1["no", "yes"] = 0.2
X_R1["no", "no"] = 0.8

X_R2 = ProbabilityMatrix(diagram, "R2")
X_R2["yes", "yes"] = 0.95
X_R2["yes", "no"] = 0.05
X_R2["no", "yes"] = 0.1
X_R2["no", "no"] = 0.9

X_R3 = ProbabilityMatrix(diagram, "R2")
X_R3["yes", "yes"] = 0.6
X_R3["yes", "no"] = 0.4
X_R3["no", "yes"] = 0.35
X_R3["no", "no"] = 0.65

add_probabilities!(diagram, "R1", X_R1)
add_probabilities!(diagram, "R2", X_R2)
add_probabilities!(diagram, "R3", X_R3)

Y_V = UtilityMatrix(diagram, "V")
Y_V["yes", "yes"] = 220
Y_V["yes","no"] = 20
Y_V["no","yes"] = 0
Y_V["no","no"] = 20
add_utilities!(diagram, "V", Y_V)

generate_diagram!(diagram)


@info("Creating the decision model.")
model = Model()
z = DecisionVariables(model, diagram, names = true, name = "z", augmented_states = true)
x_s = PathCompatibilityVariables(model, diagram, z, names = true, name = "s", probability_cut = false, augmented_states = true)
x_x = AugmentedStateVariables(model, diagram, z, x_s, names = true ,name = "x")
EV = expected_value(model, diagram, x_s, x_x = x_x)
@objective(model, Max, EV)

@info("Starting the optimization process.")
set_optimizer(model, GLPK.Optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(z)
S_probabilities = StateProbabilities(diagram, Z)
U_distribution = UtilityDistribution(diagram, Z)


@info("Printing decision strategy:")
print_decision_strategy(diagram, Z, S_probabilities, augmented_states = true)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

@info("Printing statistics")
print_statistics(U_distribution)

@info("Printing information structure")
for x in x_x
    println(x[1])
    println(value.(x[2]))
end