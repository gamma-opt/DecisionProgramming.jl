using Printf, Random, Parameters, Logging
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(11)

if isempty(ARGS)
    const N = 4
else
    const N = parse(Int, ARGS[1])
end
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

@info("Defining influence diagram parameters.")
C = L ∪ R_k ∪ F
D = A_k
V = T
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}(undef, length(C)+length(D))

@info("Defining arcs.")
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(repeat(L, N), R_k)
add_arcs(L, F)
add_arcs(R_k, A_k)
add_arcs(A_k, repeat(F, N))
add_arcs(F, T)
add_arcs(A_k, repeat(T, N))

@info("Defining states.")
S_j[L] = fill(length(L_states), length(L))
S_j[R_k] = fill(length(R_k_states), length(R_k))
S_j[A_k] = fill(length(A_k_states), length(A_k))
S_j[F] = fill(length(F_states), length(F))

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

@info("Defining InfluenceDiagram")
@time G = InfluenceDiagram(C, D, V, A, S_j)

@info("Creating probabilities.")
@time X = validate_probabilities(G, probabilities(L, R_k, A_k, F, S_j))

@info("Creating consequences.")
@time Y = validate_consequences(G, consequences(A_k, F, T, S_j))

@info("Creating path utility function.")
@time U(s) = sum(Y[v][s[G.I_j[v]]...] for v in V)

@info("Defining DecisionModel")
@time model = DecisionModel(G, X)

@info("Adding probability sum cut")
@time probability_sum_cut(model, G, X)

@info("Adding number of paths cut")
num_paths = prod(S_j[j] for j in C)
@time number_of_paths_cut(model, G, X, num_paths)

@info("Creating model objective.")
@time U⁺ = transform_affine_positive(U, S_j)
@time E = expected_value(model, U⁺, S_j)
@objective(model, Max, E)

@info("Starting the optimization process.")
optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(model)

@info("Printing decision strategy:")
print_decision_strategy(Z, G)
println()

@info("Printing state probabilities:")
probs = state_probabilities(Z, G, X)
print_state_probabilities(probs, L, L_states)
print_state_probabilities(probs, R_k, R_k_states)
print_state_probabilities(probs, A_k, A_k_states)
print_state_probabilities(probs, F, F_states)
println()

@info("Print utility distribution statistics.")
@time u, p = utility_distribution(Z, G, X, U)
include("statistics.jl")
print_stats(u, p)
