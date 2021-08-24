using Logging
using JuMP, Gurobi
using DecisionProgramming

const N = 4

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

AddNode!(diagram, ChanceNode("H1", [], ["ill", "healthy"]))
for i in 1:N-1
    # Testing result
    AddNode!(diagram, ChanceNode("T$i", ["H$i"], ["positive", "negative"]))
    # Decision to treat
    AddNode!(diagram, DecisionNode("D$i", ["T$i"], ["treat", "pass"]))
    # Cost of treatment
    AddNode!(diagram, ValueNode("C$i", ["D$i"]))
    # Health of next period
    AddNode!(diagram, ChanceNode("H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"]))
end
AddNode!(diagram, ValueNode("SP", ["H$N"]))

GenerateArcs!(diagram)
# Declare proability matrix for health nodes
X_H = zeros(2, 2, 2)
X_H[2, 2, 1] = 0.2
X_H[2, 2, 2] = 1.0 - X_H[2, 2, 1]
X_H[2, 1, 1] = 0.1
X_H[2, 1, 2] = 1.0 - X_H[2, 1, 1]
X_H[1, 2, 1] = 0.9
X_H[1, 2, 2] = 1.0 - X_H[1, 2, 1]
X_H[1, 1, 1] = 0.5
X_H[1, 1, 2] = 1.0 - X_H[1, 1, 1]

# Declare proability matrix for test results nodes
X_T = zeros(2, 2)
X_T[1, 1] = 0.8
X_T[1, 2] = 1.0 - X_T[1, 1]
X_T[2, 2] = 0.9
X_T[2, 1] = 1.0 - X_T[2, 2]

AddProbabilities!(diagram, "H1", [0.1, 0.9])
for i in 1:N-1
    # Testing result
    AddProbabilities!(diagram, "T$i", X_T)
    # Cost of treatment
    AddConsequences!(diagram, "C$i", [-100.0, 0.0])
    # Health of next period
    AddProbabilities!(diagram, "H$(i+1)", X_H)
end
# Selling price
AddConsequences!(diagram, "SP", [300.0, 1000.0])

GenerateDiagram!(diagram, positive_path_utility = true)


@info("Creating the decision model.")
model = Model()
z = DecisionVariables(model, diagram)
x_s = PathCompatibilityVariables(model, diagram, z, probability_cut = false)
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

@info("Printing statistics")
print_statistics(U_distribution)

#=
@info("State probabilities:")
sprobs = StateProbabilities(diagram.S, diagram.P, Z)
print_state_probabilities(sprobs, health)
print_state_probabilities(sprobs, test)
print_state_probabilities(sprobs, treat)

@info("Conditional state probabilities")
node = 1
for state in 1:2
    sprobs2 = StateProbabilities(diagram.S, diagram.P, Z, node, state, sprobs)
    print_state_probabilities(sprobs2, health)
    print_state_probabilities(sprobs2, test)
    print_state_probabilities(sprobs2, treat)
end
=#
