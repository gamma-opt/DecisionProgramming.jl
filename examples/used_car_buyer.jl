
using Logging
using JuMP, Gurobi
using DecisionProgramming

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

AddNode!(diagram, ChanceNode("O", [], ["lemon", "peach"]))
AddNode!(diagram, ChanceNode("R", ["O", "T"], ["no test", "lemon", "peach"]))

AddNode!(diagram, DecisionNode("T", [], ["no test", "test"]))
AddNode!(diagram, DecisionNode("A", ["R"], ["buy without guarantee", "buy with guarantee", "don't buy"]))

AddNode!(diagram, ValueNode("V1", ["T"]))
AddNode!(diagram, ValueNode("V2", ["A"]))
AddNode!(diagram, ValueNode("V3", ["O", "A"]))

GenerateArcs!(diagram)


X_R = zeros(2, 2, 3)
X_R[1, 1, :] = [1,0,0]
X_R[1, 2, :] = [0,1,0]
X_R[2, 1, :] = [1,0,0]
X_R[2, 2, :] = [0,0,1]
AddProbabilities!(diagram, "R", X_R)
AddProbabilities!(diagram, "O", [0.2, 0.8])

AddConsequences!(diagram, "V1", [0.0, -25.0])
AddConsequences!(diagram, "V2", [100.0, 40.0, 0.0])
AddConsequences!(diagram, "V3", [-200.0 0.0 0.0; -40.0 -20.0 0.0])

GenerateDiagram!(diagram)


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
