using Printf, Random
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(1)

# Parameters
N = 2 # Number of monitors
L = [1]
R_k = [k + 1 for k in 1:N]
A_k = [(N + 1) + k for k in 1:N]
F = [2*N + 2]
T = [2*N + 3]

c_k = rand(N)
fortification(a, k) = [c_k[k], 0][a]
consequence(a, k) = [-c_k[k], 0][a]

# Influence diagram parameters
C = L ∪ R_k ∪ F
D = A_k
V = T
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}()

# Arcs
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(repeat(L, N), R_k)
add_arcs(L, F)
add_arcs(R_k, A_k)
add_arcs(A_k, repeat(F, N))
add_arcs(F, T)
add_arcs(A_k, repeat(T, N))

# States
S_j = zeros(Int, length(C)+length(D))
S_j[L] = fill(2, length(L))
S_j[R_k] = fill(2, length(R_k))
S_j[A_k] = fill(2, length(A_k))
S_j[F] = fill(2, length(F))

# Probabilities
X = Dict{Int, Array{Float64}}()

for i in L
    p = zeros(S_j[i])
    p[1] = rand()
    p[2] = 1 - p[1]
    X[i] = p
end

for i in R_k
    p = zeros(S_j[1], S_j[i])
    x, y = rand(2)
    p[1, 1] = max(x, 1-y)
    p[1, 2] = 1 - p[1, 1]
    p[2, 2] = max(y, 1-y)
    p[2, 1] = 1 - p[2, 2]
    X[i] = p
end

for i in F
    p = zeros(S_j[L...], S_j[A_k]..., S_j[i])
    z, w = rand(2)
    for s in paths(S_j[A_k])
        d = exp(sum(fortification(a, k) for (k, a) in enumerate(s)))
        p[1, s..., 1] = z / d
        p[1, s..., 2] = 1 - p[1, s..., 1]
        p[2, s..., 1] = w / d
        p[2, s..., 2] = 1 - p[2, s..., 1]
    end
    X[i] = p
end

# Consequences
Y = Dict{Int, Array{Float64}}()
for v in T
    y = zeros([S_j[A_k]; S_j[F...]]...)
    for s in paths(S_j[A_k])
        c = sum(consequence(a, k) for (k, a) in enumerate(s))
        y[s..., 1] = c + 0
        y[s..., 2] = c + 100
    end
    Y[v] = y
end


# Model
diagram = InfluenceDiagram(C, D, V, A, S_j)
specs = Specs()
params = Params(diagram, X, Y)
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


# -- Results ---
I_j = diagram.I_j
S_j = diagram.S_j
πsol = model[:π]
z = model[:z]

utility(s) = sum(Y[v][s[I_j[v]]...] for v in V)

println()
println("--- Active Paths ---")
println("path | probability | utility | expected utility")
for s in paths(S_j)
    πval = value(πsol[s...])
    isapprox(πval, 0, atol=1e-3) && continue
    u = utility(s)
    eu = πval * u
    @printf("%s | %0.3f | %0.3f | %0.3f \n", s, πval, u, eu)
end

expected_utility = sum(value(πsol[s...]) * utility(s) for s in paths(S_j))
println("Expected utility: ", expected_utility)
