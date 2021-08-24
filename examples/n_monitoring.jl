using Logging, Random
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(13)

const N = 4
const c_k = rand(N)
const b = 0.03
fortification(k, a) = [c_k[k], 0][a]

@info("Creating the influence diagram.")
diagram = InfluenceDiagram()

AddNode!(diagram, ChanceNode("L", [], ["high", "low"]))

for i in 1:N
    AddNode!(diagram, ChanceNode("R$i", ["L"], ["high", "low"]))
    AddNode!(diagram, DecisionNode("A$i", ["R$i"], ["yes", "no"]))
end

AddNode!(diagram, ChanceNode("F", ["L", ["A$i" for i in 1:N]...], ["failure", "success"]))

AddNode!(diagram, ValueNode("T", ["F", ["A$i" for i in 1:N]...]))

GenerateArcs!(diagram)

X_L = [rand(), 0]
X_L[2] = 1.0 - X_L[1]
AddProbabilities!(diagram, "L", X_L)

for i in 1:N
    x, y = rand(2)
    X_R = zeros(2,2)
    X_R[1, 1] = max(x, 1-x)
    X_R[1, 2] = 1.0 - X_R[1, 1]
    X_R[2, 2] = max(y, 1-y)
    X_R[2, 1] = 1.0 - X_R[2, 2]
    AddProbabilities!(diagram, "R$i", X_R)
end

for i in [1]
    x, y = rand(2)
    X_F = zeros(2, [2 for i in 1:N]..., 2)
    for s in paths([2 for i in 1:N])
        d = exp(b * sum(fortification(k, a) for (k, a) in enumerate(s)))
        X_F[1, s..., 1] = max(x, 1-x) / d
        X_F[1, s..., 2] = 1.0 - X_F[1, s..., 1]
        X_F[2, s..., 1] = min(y, 1-y) / d
        X_F[2, s..., 2] = 1.0 - X_F[2, s..., 1]
    end
    AddProbabilities!(diagram, "F", X_F)
end

Y_T = zeros([2 for i in 1:N]..., 2)
for s in paths([2 for i in 1:N])
    cost = sum(-fortification(k, a) for (k, a) in enumerate(s))
    Y_T[s..., 1] = cost + 0
    Y_T[s..., 2] = cost + 100
end
AddConsequences!(diagram, "T", Y_T)

GenerateDiagram!(diagram)#, positive_path_utility=true)


model = Model()
z = DecisionVariables(model, diagram)
x_s = PathCompatibilityVariables(model, diagram, z)#, probability_cut = false)
EV = expected_value(model, diagram, x_s)
@objective(model, Max, EV)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(z)
U_distribution = UtilityDistribution(diagram, Z)
state_probabilities = StateProbabilities(diagram, Z)

@info("Printing decision strategy:")
print_decision_strategy(diagram, Z, state_probabilities)

@info("Printing utility distribution.")
print_utility_distribution(U_distribution)

@info("Printing statistics")
print_statistics(U_distribution)

#=
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
const b = 0.03
fortification(k, a) = [c_k[k], 0][a]

@info("Creating the influence diagram.")
S = States([
    (length(L_states), L),
    (length(R_k_states), R_k),
    (length(A_k_states), A_k),
    (length(F_states), F)
])
C = Vector{ChanceNode}()
D = Vector{DecisionNode}()
V = Vector{ValueNode}()
X = Vector{Probabilities}()
Y = Vector{Consequences}()

for j in L
    I_j = Vector{Node}()
    X_j = zeros(S[I_j]..., S[j])
    X_j[1] = rand()
    X_j[2] = 1.0 - X_j[1]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(j, X_j))
end

for j in R_k
    I_j = L
    x, y = rand(2)
    X_j = zeros(S[I_j]..., S[j])
    X_j[1, 1] = max(x, 1-x)
    X_j[1, 2] = 1.0 - X_j[1, 1]
    X_j[2, 2] = max(y, 1-y)
    X_j[2, 1] = 1.0 - X_j[2, 2]
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(j, X_j))
end

for (i, j) in zip(R_k, A_k)
    I_j = [i]
    push!(D, DecisionNode(j, I_j))
end

for j in F
    I_j = L ∪ A_k
    x, y = rand(2)
    X_j = zeros(S[I_j]..., S[j])
    for s in paths(S[A_k])
        d = exp(b * sum(fortification(k, a) for (k, a) in enumerate(s)))
        X_j[1, s..., 1] = max(x, 1-x) / d
        X_j[1, s..., 2] = 1.0 - X_j[1, s..., 1]
        X_j[2, s..., 1] = min(y, 1-y) / d
        X_j[2, s..., 2] = 1.0 - X_j[2, s..., 1]
    end
    push!(C, ChanceNode(j, I_j))
    push!(X, Probabilities(j, X_j))
end

for j in T
    I_j = A_k ∪ F
    Y_j = zeros(S[I_j]...)
    for s in paths(S[A_k])
        cost = sum(-fortification(k, a) for (k, a) in enumerate(s))
        Y_j[s..., 1] = cost + 0
        Y_j[s..., 2] = cost + 100
    end
    push!(V, ValueNode(j, I_j))
    push!(Y, Consequences(j, Y_j))
end

validate_influence_diagram(S, C, D, V)
sort!.((C, D, V, X, Y), by = x -> x.j)

P = DefaultPathProbability(C, X)
U = DefaultPathUtility(V, Y)

@info("Creating the decision model.")
U⁺ = PositivePathUtility(S, U)
model = Model()
z = DecisionVariables(model, S, D)
x_s = PathCompatibilityVariables(model, z, S, P, probability_cut = false)
EV = expected_value(model, x_s, U⁺, P)
@objective(model, Max, EV)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    () -> Gurobi.Optimizer(Gurobi.Env()),
    "IntFeasTol"      => 1e-9,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(z)

@info("Printing decision strategy:")
print_decision_strategy(S, Z)

@info("Printing state probabilities:")
sprobs = StateProbabilities(S, P, Z)
print_state_probabilities(sprobs, L)
print_state_probabilities(sprobs, R_k)
print_state_probabilities(sprobs, A_k)
print_state_probabilities(sprobs, F)

@info("Computing utility distribution.")
udist = UtilityDistribution(S, P, U, Z)

@info("Printing utility distribution.")
print_utility_distribution(udist)

@info("Printing statistics")
print_statistics(udist)
=#
