using Logging, Test, Random, JuMP, GLPK
using DecisionProgramming

rng = MersenneTwister(4)

@info "Creating the influence diagram."
C, D, V = random_diagram(rng, 4, 2, 3, 2)
S = States(rng, [2, 3], length(C) + length(D))
X = [Probabilities(rng, c, S) for c in C]
Y = [Consequences(rng, v, S) for v in V]

S, C, D, V, X, Y = validate_influence_diagram(S, C, D, V, X, Y)
P = PathProbability(C, X)
U = DefaultPathUtility(V, Y)

@info "Creating decision model."
U⁺ = PositivePathUtility(S, U)
model = DecisionModel(S, D, P; positive_path_utility=true)
probability_sum_cut(model, S, P)
number_of_paths_cut(model, S, P)

@info "Adding objectives to the model."
α = 0.2
w = 0.5
EV = expected_value(model, S, U⁺)
CVaR = conditional_value_at_risk(model, S, U⁺, α)
@objective(model, Max, w * EV + (1 - w) * CVaR)

@info "Solving the model."
optimizer = optimizer_with_attributes(GLPK.Optimizer)
set_optimizer(model, optimizer)
optimize!(model)

@info "Analyzing results."
Z = DecisionStrategy(model, D)
udist = UtilityDistribution(S, P, U, Z)
sprobs = StateProbabilities(S, P, Z)

@info "Printing results"
print_decision_strategy(S, Z)
print_utility_distribution(udist)
print_state_probabilities(sprobs, [c.j for c in C])
print_state_probabilities(sprobs, [d.j for d in D])
print_statistics(udist)
print_risk_measures(udist, [0.0, 0.05, 0.1, 0.2, 1.0])

@test true
