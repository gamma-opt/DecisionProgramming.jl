using Printf
using JuMP, Gurobi
using DecisionProgramming

health = [1, 4, 7, 10] # health of the pig
test = [2, 5, 8] # whether to test the pig
treat = [3, 6, 9] # whether to treat the pig
cost = [11, 12, 13] # treatment cost
price = [14] # sell price

C = health ∪ test
D = treat
V = cost ∪ price
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}()

# Construct arcs
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(health[1:end-1], health[2:end])
add_arcs(health[1:end-1], test)
add_arcs(treat, health[2:end])
add_arcs(treat, cost)
add_arcs(health[end], price)
n = length(test)
append!(A, (test[i] => treat[j] for i in 1:n for j in i:n))
append!(A, (treat[i] => treat[j] for i in 1:(n-1) for j in (i+1):n))

# Construct states
S_j = zeros(Int, length(C)+length(D))
S_j[health] = fill(2, length(health))
S_j[test] = fill(2, length(test))
S_j[treat] = fill(2, length(treat))

# Probabilities
X = Dict{Int, Array{Float64}}()

# h_1
begin
    i = health[1]
    p = zeros(S_j[i])
    p[1] = 0.1
    p[2] = 1.0 - 0.1
    X[i] = p
end

# h_i, i≥2
for (i, j, k) in zip(health[1:end-1], treat, health[2:end])
    p = zeros(S_j[i], S_j[j], S_j[k])
    p[2, 2, 1] = 0.2
    p[2, 2, 2] = 1.0 - 0.2
    p[2, 1, 1] = 0.1
    p[2, 1, 2] = 1.0 - 0.1
    p[1, 2, 1] = 0.9
    p[1, 2, 2] = 1.0 - 0.9
    p[1, 1, 1] = 0.5
    p[1, 1, 2] = 1.0 - 0.5
    X[k] = p
end

# t_i
for (i, j) in zip(health, test)
    p = zeros(S_j[i], S_j[j])
    p[1, 1] = 0.8
    p[1, 2] = 1.0 - 0.8
    p[2, 2] = 0.9
    p[2, 1] = 1.0 - 0.9
    X[j] = p
end

# Consequences
Y = Dict{Int, Array{Int}}()
for i in cost
    Y[i] = [1, 2]
end
for i in price
    Y[i] = [3, 4]
end

# Utilities
U = [-100, 0, 300, 1000]

# Model
diagram = InfluenceDiagram(C, D, V, A, S_j)
specs = Specs()
params = Params(X, Y, U)
model = DecisionModel(specs, diagram, params)


println("--- Optimization ---")
optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" =>    1,
    "TimeLimit"       => 2*60,
    "Heuristics"      =>  .00,  # 0% time spend on heuristics (lazy cuts need to be re-added if heuristic is used in root node)
    "BranchDir"       =>    1,  # -1 ... 1: 1 = branch up
    "VarBranch"       =>    0,  # -1 ... 3: 0 = pseudo costs, 3 = Strong branching
    "Cuts"            =>    0,  # -1 ... 3: 0 = all cuts off (use lazy cuts instead that exploit problem structure)
    "DisplayInterval"   => 5
)
set_optimizer(model, optimizer)
optimize!(model)

println()
println("--- Active Paths ---")
println("path | probability | utility | expected utility")

I_j = diagram.I_j
S_j = diagram.S_j

utility(s) = sum(U[Y[v][s[I_j[v]]...]] for v in V)
π = model[:π]

for s in paths(S_j)
    πval = value(π[s...])
    isapprox(πval, 0, atol=1e-3) && continue
    u = utility(s)
    eu = πval * u
    @printf("%s | %0.3f | %0.3f | %0.3f \n", s, πval, u, eu)
end

expected_utility = sum(value(π[s...]) * utility(s) for s in paths(S_j))
println("Expected utility: ", expected_utility)
