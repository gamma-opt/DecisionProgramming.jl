using Test, Random
using DecisionProgramming

Random.seed!(111)

diagram = random_influence_diagram(3, 3, 3, 2, [2])
params = random_params(diagram)
model = DecisionModel(diagram, params)

@test true
