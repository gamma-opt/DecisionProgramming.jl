using Random, Parameters, Printf, JuMP, Gurobi
using DecisionProgramming

Random.seed!(111)

G = random_influence_diagram(5, 3, 2, 2, [2, 3])
X = random_probabilities(G)
Y = random_consequences(G)
U(s) = sum(Y[v][s[G.I_j[v]]...] for v in G.V)

model = DecisionModel(G, X)
U⁺ = transform_affine_positive(G, U)
EV = expected_value(model, G, U⁺)
ES = conditional_value_at_risk(model, G, U⁺, 0.2)

# @objective(model, Max, EV)
@objective(model, Max, ES)
# w = 0.5
# @objective(model, Max, w * EV + (1 - w) * ES)

optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(model)

# @info("Printing decision strategy:")
# print_decision_strategy(G, Z)

@info("Printing state probabilities:")
probs = StateProbabilities(G, X, Z)
print_state_probabilities(probs, G.C)
print_state_probabilities(probs, G.D)
println()

@info("Computing utility distribution.")
@time udist = UtilityDistribution(G, X, Z, U)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)

@info("Printing risk measures")
αs = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
print_risk_measures(udist, αs)
