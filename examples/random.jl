using Random, Parameters, Printf, JuMP, Gurobi
using DecisionProgramming

rng = MersenneTwister(111)

C, D, V = random_diagram(rng, 5, 3, 2, 3)
S = States(rng, [2, 3], length(C) + length(D))
X = [Probabilities(rng, c, S; n_inactive=0) for c in C]
Y = [Consequences(rng, v, S, low=-1.0, high=1.5) for v in V]

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

U⁺ = PositivePathUtility(S, U)
model = DecisionModel(S, D, P; positive_path_utility=true)
# probability_cut(model, S, P)

α = 0.1
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

Z = DecisionStrategy(model, D)

@info("Printing decision strategy:")
print_decision_strategy(S, Z)

@info("Printing state probabilities:")
sprobs = StateProbabilities(S, P, Z)
print_state_probabilities(sprobs, [c.j for c in C])
print_state_probabilities(sprobs, [d.j for d in D])

@info("Computing utility distribution.")
udist = UtilityDistribution(S, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)

@info("Printing risk measures")
print_risk_measures(udist, [α])
