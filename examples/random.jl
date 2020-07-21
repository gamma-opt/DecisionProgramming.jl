using Random, Parameters, Printf
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(111)

G = random_influence_diagram(5, 3, 2, 2, [2, 3])
X = random_probabilities(G)
Y = random_consequences(G)
model = DecisionModel(G, X)
@unpack V, S_j, I_j = G
@time U(s) = sum(Y[v][s[I_j[v]]...] for v in V)
@time U⁺ = transform_affine_positive(U, S_j)
@time E = expected_value(model, U⁺, S_j)
# @objective(model, Max, E)
α = 0.2
@time CVaR = value_at_risk(model, U⁺, S_j, α)
@objective(model, Max, CVaR)
# w = 0.8
# @objective(model, Max, w * E + (1 - w) * CVaR)

optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
Z = DecisionStrategy(model)

# @info("Printing decision strategy:")
# print_decision_strategy(z, diagram)

@info("Printing state probabilities:")
probs = state_probabilities(Z, G, X)
print_state_probabilities(probs, G.C, [])
print_state_probabilities(probs, G.D, [])
println()

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
