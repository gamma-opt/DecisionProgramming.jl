using Random, Parameters
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(111)

diagram = random_influence_diagram(5, 3, 2, 2, [2, 3])
params = random_params(diagram)
model = DecisionModel(diagram, params)
@unpack V, S_j, I_j = diagram
@unpack Y = params
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
z = DecisionStrategy(model)

# @info("Printing decision strategy:")
# print_decision_strategy(z, diagram)

@info("Printing state probabilities:")
probs = state_probabilities(z, diagram, params)
print_state_probabilities(probs, diagram.C, [])
print_state_probabilities(probs, diagram.D, [])
println()

@info("Create results directory.")
using Dates
directory = joinpath("random", string(now()))
if !ispath(directory)
    mkpath(directory)
end

@info("Plot the utility distributions.")
using Plots
@time u, p = utility_distribution(z, diagram, params, U)

p1 = plot(u, p,
    linewidth=0,
    markershape=:circle,
    ylims=(0, maximum(p) + 0.15),
    label="Distribution",
    legend=:topleft)
savefig(p1, joinpath(directory, "utility-distribution.svg"))

p2 = plot(u, cumsum(p),
    linestyle=:dash,
    markershape=:circle,
    ylims=(0, 1 + 0.15),
    label="Cumulative distribution",
    legend=:topleft)
savefig(p2, joinpath(directory, "cumulative-utility-distribution.svg"))
