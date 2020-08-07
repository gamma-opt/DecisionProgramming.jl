using Random, Parameters, Printf, JuMP, Gurobi
using DecisionProgramming

rng = MersenneTwister(111)

C, D, V = random_diagram(rng, 5, 3, 2, 2)
S = States(rng, [2, 3], length(C) + length(D))
X = [Probabilities(rng, c, S) for c in C]
Y = [Consequences(rng, v, S) for v in V]

S, C, D, V, X, Y = validate_influence_diagram(S, C, D, V, X, Y)
P = PathProbability(C, X)
U = DefaultPathUtility(V, Y)

U⁺ = PositivePathUtility(S, U)
model = DecisionModel(S, D, P; positive_path_utility=true)
# probability_sum_cut(model, S, P)
# number_of_paths_cut(model, S, P)

α = 0.2
w = 0.5
EV = expected_value(model, S, U⁺)
CVaR = conditional_value_at_risk(model, S, U⁺, α)
@objective(model, Max, w * EV + (1 - w) * CVaR)

optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = GlobalDecisionStrategy(model, D)

@info("Printing decision strategy:")
print_decision_strategy(S, Z)

@info("Printing state probabilities:")
sprobs = StateProbabilities(S, P, Z)
print_state_probabilities(sprobs, [c.j for c in C])
print_state_probabilities(sprobs, [d.j for d in D])

@info("Computing utility distribution.")
@time udist = UtilityDistribution(S, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)

@info("Printing risk measures")
αs = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
print_risk_measures(udist, αs)
