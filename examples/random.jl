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
@objective(model, Max, E)

optimizer = optimizer_with_attributes(
    Gurobi.Optimizer,
    "IntFeasTol"      => 1e-9,
    "LazyConstraints" => 1,
)
set_optimizer(model, optimizer)
optimize!(model)

@info("Extracting results.")
round_int(z) = Int(round(z))
z = Dict(i => round_int.(value.(model[:z][i])) for i in diagram.D)

@info("Printing results")
print_results(z, diagram, params, U)

@info("Printing decision strategy:")
print_decision_strategy(z, diagram)

@info("Printing state probabilities:")
probs = state_probabilities(z, diagram, params)
print_state_probabilities(probs, diagram.C, [])
print_state_probabilities(probs, diagram.D, [])
println()

@info("Plot the cumulative distribution.")
using Plots
@time x, y = utility_distribution(z, diagram, params, U)
p = plot(x, y,
    linestyle=:dash,
    markershape=:circle,
    ylims=(0, 1.1),
    label="Distribution",
    legend=:topleft)
plot!(p, x, cumsum(y),
    linestyle=:dash,
    markershape=:circle,
    label="Cumulative distribution")
savefig(p, "random-utility-distribution.svg")
