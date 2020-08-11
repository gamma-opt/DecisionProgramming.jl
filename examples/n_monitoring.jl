using Printf, Random, Logging, Parameters, JuMP, Gurobi
using DecisionProgramming

Random.seed!(11)

const N = 4
const L = [1]
const R_k = [k + 1 for k in 1:N]
const A_k = [(N + 1) + k for k in 1:N]
const F = [2*N + 2]
const T = [2*N + 3]
const L_states = ["high", "low"]
const R_k_states = ["high", "low"]
const A_k_states = ["yes", "no"]
const F_states = ["failure", "success"]
const c_k = rand(N)
fortification(k, a) = [c_k[k], 0][a]
consequence(k, a) = [-c_k[k], 0][a]

C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()

X = Vector{Probabilities}()
Y = Vector{Consequences}()

S_j = Vector{State}(undef, length(L) + length(R_k) + length(A_k) + length(F))
S_j[L] = fill(length(L_states), length(L))
S_j[R_k] = fill(length(R_k_states), length(R_k))
S_j[A_k] = fill(length(A_k_states), length(A_k))
S_j[F] = fill(length(F_states), length(F))
S = States(S_j)

for j in L
    I_j = Vector{Node}()
    X_j = zeros(S_j[j])
    X_j[1] = rand()
    X_j[2] = 1.0 - X_j[1]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end

for j in R_k
    I_j = L
    x, y = rand(2)
    X_j = zeros(S_j[I_j]..., S_j[j])
    X_j[1, 1] = max(x, 1-x)
    X_j[1, 2] = 1.0 - X_j[1, 1]
    X_j[2, 2] = max(y, 1-y)
    X_j[2, 1] = 1.0 - X_j[2, 2]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end

for (i, j) in zip(R_k, A_k)
    I_j = [i]
    push!(D, DecisionNode(j, I_j))
end

for j in F
    I_j = L ∪ A_k
    x, y = rand(2)
    X_j = zeros(S_j[I_j]..., S_j[j])
    for s in paths(S_j[A_k])
        d = exp(sum(fortification(k, a) for (k, a) in enumerate(s)))
        X_j[1, s..., 1] = max(x, 1-x) / d
        X_j[1, s..., 2] = 1.0 - X_j[1, s..., 1]
        X_j[2, s..., 1] = min(y, 1-y) / d
        X_j[2, s..., 2] = 1.0 - X_j[2, s..., 1]
    end
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(X_j))
end

for j in T
    I_j = A_k ∪ F
    Y_j = zeros(S_j[I_j]...)
    for s in paths(S_j[A_k])
        c = sum(consequence(k, a) for (k, a) in enumerate(s))
        Y_j[s..., 1] = c + 0
        Y_j[s..., 2] = c + 100
    end
    push!(V, ValueNode(j, I_j))
    push!(Y, Consequences(Y_j))
end

@info("Validate influence diagram.")
validate_influence_diagram(S, C, D, V)
s_c = sortperm([c.j for c in C])
s_d = sortperm([d.j for d in D])
s_v = sortperm([v.j for v in V])
C = C[s_c]
D = D[s_d]
V = V[s_v]
X = X[s_c]
Y = Y[s_v]

@info("Creating path probability.")
P = DefaultPathProbability(C, X)

@info("Creating path utility.")
U = DefaultPathUtility(V, Y)

@info("Defining DecisionModel")
U⁺ = PositivePathUtility(S, U)
@time model = DecisionModel(S, D, P; positive_path_utility=true)

@info("Adding probability sum cut")
@time probability_sum_cut(model, S, P)

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

@info("Printing state probabilities:")
sprobs = StateProbabilities(S, P, Z)
print_state_probabilities(sprobs, L)
print_state_probabilities(sprobs, R_k)
print_state_probabilities(sprobs, A_k)
print_state_probabilities(sprobs, F)

@info("Computing utility distribution.")
@time udist = UtilityDistribution(S, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)

@info("Printing risk measures")
αs = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
print_risk_measures(udist, αs)
