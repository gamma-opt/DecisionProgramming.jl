using Test, Random, Parameters
using JuMP
using DecisionProgramming

Random.seed!(111)

diagram = random_influence_diagram(3, 3, 3, 2, [2])
params = random_params(diagram)
model = DecisionModel(diagram, params)
probability_sum_cut(model, diagram, params)
num_paths = prod(diagram.S_j[j] for j in diagram.C)
number_of_paths_cut(model, diagram, params, num_paths)
@unpack V, S_j, I_j = diagram
@unpack Y = params
@time U(s) = sum(Y[v][s[I_j[v]]...] for v in V)
@time U⁺ = transform_affine_positive(U, S_j)
@time E = expected_value(model, U⁺, S_j)
@objective(model, Max, E)

@test true
