using Test, Random, Parameters
using JuMP
using DecisionProgramming

Random.seed!(111)

G = random_influence_diagram(3, 3, 3, 2, [2])
X = random_probabilities(G)
Y = random_consequences(G)
model = DecisionModel(G, X)
probability_sum_cut(model, G, X)
num_paths = prod(G.S_j[j] for j in G.C)
number_of_paths_cut(model, G, X, num_paths)
@unpack V, S_j, I_j = G
@time U(s) = sum(Y[v][s[I_j[v]]...] for v in V)
@time U⁺ = transform_affine_positive(U, S_j)
@time E = expected_value(model, U⁺, S_j)
@objective(model, Max, E)

@test true
