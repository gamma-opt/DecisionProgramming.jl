using Random
using JuMP, Gurobi
using DecisionProgramming

Random.seed!(111)

diagram = random_influence_diagram(5, 3, 2, 2, [2, 3])
params = random_params(diagram)
model = DecisionModel(diagram, params)
E = expected_value(model, diagram, params)
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
print_results(z, diagram, params)

@info("Printing decision strategy:")
print_decision_strategy(z, diagram)

@info("Printing state probabilities:")
probs = state_probabilities(z, diagram, params)
print_state_probabilities(probs, diagram.C, [])
print_state_probabilities(probs, diagram.D, [])
println()
