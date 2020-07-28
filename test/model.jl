using Logging, Test, Random, JuMP, GLPK
using DecisionProgramming

Random.seed!(4)

@info "Creating influence diagram, probabilities and consequences."
G = random_influence_diagram(4, 2, 3, 2, [2])
X = validate_probabilities(G, random_probabilities(G))
Y = validate_consequences(G, random_consequences(G))
U(s) = sum(Y[v][s[G.I_j[v]]...] for v in G.V)

@info "Creating decision model."
model = DecisionModel(G, X)
probability_sum_cut(model, G, X)
number_of_paths_cut(model, G, X)

@info "Adding objectives to the model."
U⁺ = transform_affine_positive(G, U)
EV = expected_value(model, G, U⁺)
ES = conditional_value_at_risk(model, G, U⁺, 0.2)
w = 0.5
@objective(model, Max, w * EV + (1 - w) * ES)

@info "Solving the model."
optimizer = optimizer_with_attributes(GLPK.Optimizer)
set_optimizer(model, optimizer)
optimize!(model)

@info "Analyzing results."
Z = DecisionStrategy(model)
udist = UtilityDistribution(G, X, Z, U)
sprobs = StateProbabilities(G, X, Z)

@info "Printing results"
print_decision_strategy(G, Z)
print_utility_distribution(udist)
print_state_probabilities(sprobs, G.C)
print_state_probabilities(sprobs, G.D)
print_statistics(udist)
print_risk_measures(udist, [0.0, 0.05, 0.1, 0.2, 1.0])

@test true
