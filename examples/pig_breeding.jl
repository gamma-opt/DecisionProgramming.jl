using Printf, Parameters
using JuMP, Gurobi
using DecisionProgramming

if isempty(ARGS)
    const N = 4
else
    const N = parse(Int, ARGS[1])
end
const health = [3*k - 2 for k in 1:N]
const test = [3*k - 1 for k in 1:(N-1)]
const treat = [3*k for k in 1:(N-1)]
const cost = [(3*N - 2) + k for k in 1:(N-1)]
const price = [(3*N - 2) + N]
const health_states = ["ill", "healthy"]
const test_states = ["positive", "negative"]
const treat_states = ["treat", "pass"]
const no_forgetting = false

@info("Defining influence diagram parameters.")
C = health ∪ test
D = treat
V = cost ∪ price
A = Vector{Pair{Int, Int}}()
S_j = Vector{Int}(undef, length(C) + length(D))

@info("Defining arcs.")
add_arcs(from, to) = append!(A, (i => j for (i, j) in zip(from, to)))
add_arcs(health[1:end-1], health[2:end])
add_arcs(health[1:end-1], test)
add_arcs(treat, health[2:end])
add_arcs(treat, cost)
add_arcs(health[end], price)
if no_forgetting
    append!(A, (test[k] => treat[k2] for k in 1:(N-1) for k2 in k:(N-1)))
    append!(A, (treat[k] => treat[k2] for k in 1:((N-1)-1) for k2 in (k+1):(N-1)))
else
    add_arcs(test, treat)
end

@info("Defining states.")
S_j[health] = fill(length(health_states), length(health))
S_j[test] = fill(length(test_states), length(test))
S_j[treat] = fill(length(treat_states), length(treat))

function probabilities(health, treat, test, S_j)
    X = Dict{Int, Array{Float64}}()

    # h_1
    begin
        i = health[1]
        p = zeros(S_j[i])
        p[1] = 0.1
        p[2] = 1.0 - p[1]
        X[i] = p
    end

    # h_i, i≥2
    for (i, j, k) in zip(health[1:end-1], treat, health[2:end])
        p = zeros(S_j[i], S_j[j], S_j[k])
        p[2, 2, 1] = 0.2
        p[2, 2, 2] = 1.0 - p[2, 2, 1]
        p[2, 1, 1] = 0.1
        p[2, 1, 2] = 1.0 - p[2, 1, 1]
        p[1, 2, 1] = 0.9
        p[1, 2, 2] = 1.0 - p[1, 2, 1]
        p[1, 1, 1] = 0.5
        p[1, 1, 2] = 1.0 - p[1, 1, 1]
        X[k] = p
    end

    # t_i
    for (i, j) in zip(health, test)
        p = zeros(S_j[i], S_j[j])
        p[1, 1] = 0.8
        p[1, 2] = 1.0 - p[1, 1]
        p[2, 2] = 0.9
        p[2, 1] = 1.0 - p[2, 2]
        X[j] = p
    end
    return X
end

function consequences(cost, price)
    Y = Dict{Int, Array{Float64}}()
    for i in cost
        Y[i] = [-100, 0]
    end
    for i in price
        Y[i] = [300, 1000]
    end
    return Y
end

@info("Defining InfluenceDiagram")
@time G = InfluenceDiagram(C, D, V, A, S_j)

@info("Creating probabilities.")
@time X = validate_probabilities(G, probabilities(health, treat, test, S_j))

@info("Creating consequences.")
@time Y = validate_consequences(G, consequences(cost, price))

@info("Defining DecisionModel")
@time model = DecisionModel(G, X)

@info("Adding number of paths cut")
num_paths = prod(S_j[j] for j in C)
@time number_of_paths_cut(model, G, X, num_paths)

@info("Creating model objective.")
I_j = G.I_j
@time U(s) = sum(Y[v][s[I_j[v]]...] for v in V)
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

@info("State probabilities:")
probs = state_probabilities(Z, G, X)
print_state_probabilities(probs, health, health_states)
print_state_probabilities(probs, test, test_states)
print_state_probabilities(probs, treat, treat_states)
println()

# # Conditional state probabilities when pig is treat or not treated.
# for node in treat
#     for state in 1:2
#         fixed = Dict(node => state)
#         prior = probs[node][state]
#         (isapprox(prior, 0, atol=1e-4) | isapprox(prior, 1, atol=1e-4)) && continue
#         @info("Conditional state probabilities")
#         probs2 = state_probabilities(z, G, X, prior, fixed)
#         print_state_probabilities(probs2, health, health_states, fixed)
#         print_state_probabilities(probs2, test, test_states, fixed)
#         print_state_probabilities(probs2, treat, treat_states, fixed)
#         println()
#     end
# end

@info("Print utility distribution statistics.")
@time u, p = utility_distribution(Z, G, X, U)

using StatsBase
using StatsBase.Statistics
w = ProbabilityWeights(p)
println("Mean: ", mean(u, w))
println("Std: ", std(u, w, corrected=false))
println("Skewness: ", skewness(u, w))
println("Kurtosis: ", kurtosis(u, w))
println("Value-at-risk (VaR)")
println("α | VaR_α(Z)")
for α in [0.01, 0.05, 0.1, 0.2]
    @printf("%.2f | %.2f \n", α, quantile(u, w, α))
end
