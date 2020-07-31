using Random, Parameters, Printf, JuMP, Gurobi
using DecisionProgramming

rng = MersenneTwister(111)

G = InfluenceDiagram(rng, 5, 3, 2, 2, [2, 3])
X = Probabilities(rng, G)
Y = Consequences(rng, G)
P = PathProbability(G, X)
U = PathUtility(G, Y)

model = DecisionModel(G, P)
# probability_sum_cut(model, P)
# number_of_paths_cut(model, G, P)

α = 0.2
w = 0.5
EV = expected_value(model, G, U)
ES = conditional_value_at_risk(model, G, U, α)
@objective(model, Max, w * EV + (1 - w) * ES)

optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(model)

@info("Printing decision strategy:")
print_decision_strategy(G, Z)

@info("Printing state probabilities:")
probs = StateProbabilities(G, P, Z)
print_state_probabilities(probs, G.C)
print_state_probabilities(probs, G.D)
println()

@info("Computing utility distribution.")
@time udist = UtilityDistribution(G, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)

@info("Printing risk measures")
αs = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
print_risk_measures(udist, αs)
