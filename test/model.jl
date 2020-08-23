using Logging, Test, Random, JuMP
using DecisionProgramming

rng = MersenneTwister(4)

@info "Creating the influence diagram."
C, D, V = random_diagram(rng, 5, 3, 3, 3)
S = States(rng, [2, 3], length(C) + length(D))
X = [Probabilities(rng, c, S; n_inactive=rand(rng, 0:1)) for c in C]
Y = [Consequences(rng, v, S; low=-1.0, high=1.0) for v in V]

validate_influence_diagram(S, C, D, V)
s_c = sortperm([c.j for c in C])
s_d = sortperm([d.j for d in D])
s_v = sortperm([v.j for v in V])
C = C[s_c]
D = D[s_d]
V = V[s_v]
X = X[s_c]
Y = Y[s_v]

P = DefaultPathProbability(C, X)
U = DefaultPathUtility(V, Y)

@info "Creating decision model."
U⁺ = PositivePathUtility(S, U)
model = DecisionModel(S, D, P; positive_path_utility=true)
probability_cut(model, S, P)
# active_paths_cut(model, S, P)

@info "Adding objectives to the model."
α = 0.2
w = 0.5
EV = expected_value(model, S, U⁺)
CVaR = conditional_value_at_risk(model, S, U⁺, α)
@objective(model, Max, w * EV + (1 - w) * CVaR)

@info("Creating random decision strategy")
Z_j = [LocalDecisionStrategy(rng, d, S) for d in D]
Z = DecisionStrategy(D, Z_j)

@info "Analyzing results."
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
