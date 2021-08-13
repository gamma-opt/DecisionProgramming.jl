using Logging
using JuMP, Gurobi
using DecisionProgramming

const N = 4


@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

AddChanceNode!(diagram, "H1", Vector{Name}(), ["ill", "healthy"], [0.1, 0.9])

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

for i in 1:N-1
    # Testing result
    AddChanceNode!(diagram, "T$i", ["H$i"], ["positive", "negative"], X_T)
    # Decision to treat
    AddDecisionNode!(diagram, "D$i", ["T$i"], ["treat", "pass"])
    # Cost of treatment
    AddValueNode!(diagram, "C$i", ["D$i"], [-100.0, 0.0])
    # Health of next period
    AddChanceNode!(diagram, "H$(i+1)", ["H$(i)", "D$(i)"], ["ill", "healthy"], X_H)
end

# Selling price
AddValueNode!(diagram, "SP", ["H$N"], [300.0, 1000.0])

GenerateDiagram!(diagram)

@info("Creating the decision model.")
#U⁺ = PositivePathUtility(S, U)
model = Model()
z = DecisionVariables(model, diagram)
x_s = PathCompatibilityVariables(model, diagram, z, probability_cut = true)
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

@info("Printing decision strategy:")
print_decision_strategy(diagram, Z)

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
@info("Computing utility distribution.")
udist = UtilityDistribution(diagram.S, diagram.P, diagram.U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)
