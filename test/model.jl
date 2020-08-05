using Logging, Test, Random, JuMP, GLPK
using DecisionProgramming

rng = MersenneTwister(4)

@info "Creating influence diagram, probabilities and consequences."
G = InfluenceDiagram(rng, 4, 2, 3, 2, [2])
X = Probabilities(rng, G)
Y = Consequences(rng, G)
P = PathProbability(G, X)
U = PathUtility(G, Y)

@info "Creating decision model."
U⁺ = PositivePathUtility(U)
model = DecisionModel(G, P; positive_path_utility=true)
probability_sum_cut(model, P)
number_of_paths_cut(model, G, P)

@info "Adding objectives to the model."
α = 0.2
w = 0.5
EV = expected_value(model, G, U⁺)
CVaR = conditional_value_at_risk(model, G, U⁺, α)
@objective(model, Max, w * EV + (1 - w) * CVaR)

@info "Solving the model."
optimizer = optimizer_with_attributes(GLPK.Optimizer)
set_optimizer(model, optimizer)
optimize!(model)

@info "Analyzing results."
Z = DecisionStrategy(model)
udist = UtilityDistribution(G, P, U, Z)
sprobs = StateProbabilities(G, P, Z)

@info "Printing results"
print_decision_strategy(G, Z)
print_utility_distribution(udist)
print_state_probabilities(sprobs, G.C)
print_state_probabilities(sprobs, G.D)
print_statistics(udist)
print_risk_measures(udist, [0.0, 0.05, 0.1, 0.2, 1.0])

@test true
