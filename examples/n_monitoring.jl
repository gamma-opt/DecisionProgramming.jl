using Printf, Random
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(11)

# Parameters
const N = 4
const L = [1]
const R_k = [k + 1 for k in 1:N]
const A_k = [(N + 1) + k for k in 1:N]
const F = [2*N + 2]
const T = [2*N + 3]
const c_k = rand(N)
fortification(k, a) = [c_k[k], 0][a]
consequence(k, a) = [-c_k[k], 0][a]

# Influence diagram parameters
C = L ∪ R_k ∪ F
D = A_k
V = T
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}(undef, length(C)+length(D))

# Arcs
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(repeat(L, N), R_k)
add_arcs(L, F)
add_arcs(R_k, A_k)
add_arcs(A_k, repeat(F, N))
add_arcs(F, T)
add_arcs(A_k, repeat(T, N))

# States
S_j[L] = fill(2, length(L))
S_j[R_k] = fill(2, length(R_k))
S_j[A_k] = fill(2, length(A_k))
S_j[F] = fill(2, length(F))

# Probabilities
function probabilities(L, R_k, A_k, F, S_j)
    X = Dict{Int, Array{Float64}}()

    # Probability of high=1 / low=2 load
    for i in L
        p = zeros(S_j[i])
        p[1] = rand()
        p[2] = 1.0 - p[1]
        X[i] = p
    end

    # Probabilities of high=1 / low=2 reports
    for i in R_k
        p = zeros(S_j[L...], S_j[i])
        x, y = rand(2)
        p[1, 1] = max(x, 1-y)
        p[1, 2] = 1.0 - p[1, 1]
        p[2, 2] = max(y, 1-y)
        p[2, 1] = 1.0 - p[2, 2]
        X[i] = p
    end

    # Probabilities of failure=1 / success=2
    for i in F
        p = zeros(S_j[L...], S_j[A_k]..., S_j[i])
        z, w = rand(2)
        for s in paths(S_j[A_k])
            d = exp(sum(fortification(k, a) for (k, a) in enumerate(s)))
            p[1, s..., 1] = z / d
            p[1, s..., 2] = 1.0 - p[1, s..., 1]
            p[2, s..., 1] = w / d
            p[2, s..., 2] = 1.0 - p[2, s..., 1]
        end
        X[i] = p
    end
    return X
end

# Consequences
function consequences(A_k, F, T, S_j)
    Y = Dict{Int, Array{Float64}}()
    for v in T
        y = zeros([S_j[A_k]; S_j[F...]]...)
        for s in paths(S_j[A_k])
            c = sum(consequence(k, a) for (k, a) in enumerate(s))
            y[s..., 1] = c + 0
            y[s..., 2] = c + 100
        end
        Y[v] = y
    end
    return Y
end

X = @time probabilities(L, R_k, A_k, F, S_j)
Y = @time consequences(A_k, F, T, S_j)


# Model
specs = Specs(probability_sum_cut=true)
diagram = @time InfluenceDiagram(C, D, V, A, S_j)
params = @time Params(diagram, X, Y)
model = @time DecisionModel(specs, diagram, params)

println("--- Optimization ---")
optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" =>    1,
)
set_optimizer(model, optimizer)
optimize!(model)

include("analysis.jl")

πval = value.(model[:π])
print_results(πval, diagram, params; πtol=0.1)

println("State probabilities:")
probs = state_probabilities(πval, diagram)
print_state_probabilities(probs, L, ["high", "low"])
print_state_probabilities(probs, R_k, ["high", "low"])
print_state_probabilities(probs, A_k, ["yes", "no"])
print_state_probabilities(probs, F, ["failure", "success"])
println()
