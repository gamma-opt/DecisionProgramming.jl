using Test, Random
using JuMP
using DecisionProgramming

Random.seed!(111)

diagram = random_influence_diagram(3, 3, 3, 2, [2])
params = random_params(diagram)
model = DecisionModel(diagram, params)
probability_sum_cut(model, diagram, params)
num_paths = prod(diagram.S_j[j] for j in diagram.C)
number_of_paths_cut(model, diagram, params, num_paths)
E = expected_value(model, diagram, params)
@objective(model, Max, E)

@test true
