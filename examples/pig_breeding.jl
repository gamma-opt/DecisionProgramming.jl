using Printf, Parameters, JuMP, Gurobi
using DecisionProgramming

const N = 4
const health = [3*k - 2 for k in 1:N]
const test = [3*k - 1 for k in 1:(N-1)]
const treat = [3*k for k in 1:(N-1)]
const cost = [(3*N - 2) + k for k in 1:(N-1)]
const price = [(3*N - 2) + N]
const health_states = ["ill", "healthy"]
const test_states = ["positive", "negative"]
const treat_states = ["treat", "pass"]

C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()

X = Vector{Probabilities}()
Y = Vector{Consequences}()

S_j = Vector{State}(undef, length(health) + length(test) + length(treat))
S_j[health] = fill(length(health_states), length(health))
S_j[test] = fill(length(test_states), length(test))
S_j[treat] = fill(length(treat_states), length(treat))
S = States(S_j)

for j in health[[1]]
    I_j = Vector{Node}()
    X_j = zeros(S_j[j])
    X_j[1] = 0.1
    X_j[2] = 1.0 - X_j[1]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end

for (i, j) in zip(health, test)
    I_j = [i]
    X_j = zeros(S_j[I_j]..., S_j[j])
    X_j[1, 1] = 0.8
    X_j[1, 2] = 1.0 - X_j[1, 1]
    X_j[2, 2] = 0.9
    X_j[2, 1] = 1.0 - X_j[2, 2]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end

for (i, j) in zip(test, treat)
    I_j = [i]
    push!(D, DecisionNode(j, I_j))
end

for (i, k, j) in zip(health[1:end-1], treat, health[2:end])
    I_j = [i, k]
    X_j = zeros(S_j[I_j]..., S_j[j])
    X_j[2, 2, 1] = 0.2
    X_j[2, 2, 2] = 1.0 - X_j[2, 2, 1]
    X_j[2, 1, 1] = 0.1
    X_j[2, 1, 2] = 1.0 - X_j[2, 1, 1]
    X_j[1, 2, 1] = 0.9
    X_j[1, 2, 2] = 1.0 - X_j[1, 2, 1]
    X_j[1, 1, 1] = 0.5
    X_j[1, 1, 2] = 1.0 - X_j[1, 1, 1]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end

for (i, j) in zip(treat, cost)
    I_j = [i]
    Y_j = zeros(S_j[I_j]...)
    Y_j[1] = -100
    Y_j[2] = 0
    push!(V, ValueNode(j, I_j))
    push!(Y, Consequences(Y_j))
end

for (i, j) in zip(health[end], price)
    I_j = [i]
    Y_j = zeros(S_j[I_j]...)
    Y_j[1] = 300
    Y_j[2] = 1000
    push!(V, ValueNode(j, I_j))
    push!(Y, Consequences(Y_j))
end

@info("Validate influence diagram.")
S, C, D, V, X, Y = validate_influence_diagram(S, C, D, V, X, Y)

@info("Creating path probability.")
P = DefaultPathProbability(C, X)

@info("Creating path utility.")
U = DefaultPathUtility(V, Y)

@info("Defining DecisionModel")
U⁺ = PositivePathUtility(S, U)
@time model = DecisionModel(S, D, P; positive_path_utility=true)

@info("Adding number of paths cut")
@time number_of_paths_cut(model, S, P)

@info("Creating model objective.")
@time EV = expected_value(model, S, U⁺)
@objective(model, Max, EV)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(model, D)

@info("Printing decision strategy:")
print_decision_strategy(S, Z)

@info("State probabilities:")
sprobs = StateProbabilities(S, P, Z)
print_state_probabilities(sprobs, health)
print_state_probabilities(sprobs, test)
print_state_probabilities(sprobs, treat)

@info("Conditional state probabilities")
node = 1
for state in 1:2
    sprobs2 = StateProbabilities(S, P, Z, node, state, sprobs)
    print_state_probabilities(sprobs2, health)
    print_state_probabilities(sprobs2, test)
    print_state_probabilities(sprobs2, treat)
end

@info("Computing utility distribution.")
@time udist = UtilityDistribution(S, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)

@info("Printing risk measures")
αs = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
print_risk_measures(udist, αs)
