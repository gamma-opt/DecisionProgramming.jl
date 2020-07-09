using Test, Random
using DecisionProgramming

Random.seed!(111)

specs = Specs()
diagram = random_influence_diagram(3, 3, 3, 2, [2])
params = random_params(diagram)
model = DecisionModel(specs, diagram, params)

@test true
