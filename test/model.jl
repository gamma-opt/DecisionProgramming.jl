using Logging, Test, Random, Parameters, JuMP, GLPK
using DecisionProgramming

Random.seed!(111)

@info "Creating influence diagram, probabilities and consequences."
G = random_influence_diagram(3, 3, 3, 2, [2])
X = validate_probabilities(G, random_probabilities(G))
Y = validate_consequences(G, random_consequences(G))

@info "Creating decision model."
model = DecisionModel(G, X)
probability_sum_cut(model, G, X)
num_paths = prod(G.S_j[j] for j in G.C)
number_of_paths_cut(model, G, X, num_paths)

@info "Adding objectives to the model."
@unpack V, S_j, I_j = G
U(s) = sum(Y[v][s[I_j[v]]...] for v in V)
U⁺ = transform_affine_positive(U, S_j)
EV = expected_value(model, U⁺, S_j)
ES = conditional_value_at_risk(model, U⁺, S_j, 0.2)
w = 0.5
@objective(model, Max, w * EV + (1 - w) * ES)

@info "Solving the model."
optimizer = optimizer_with_attributes(GLPK.Optimizer)
set_optimizer(model, optimizer)
optimize!(model)

@info "Analyzing results."
Z = DecisionStrategy(model)
u, p = utility_distribution(Z, G, X, U)
probs = state_probabilities(Z, G, X)

@test true
