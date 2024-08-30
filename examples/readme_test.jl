using DecisionProgramming

diagram = InfluenceDiagram()

add_node!(diagram, DecisionNode("A", [], ["a", "b"]))
add_node!(diagram, DecisionNode("D", ["B", "C"], ["k", "l"]))
add_node!(diagram, ChanceNode("B", ["A"], ["x", "y"]))
add_node!(diagram, ChanceNode("C", ["A"], ["v", "w"]))
add_node!(diagram, ValueNode("V", ["D"]))

generate_arcs!(diagram)

add_probabilities!(diagram, "B", [0.4 0.6; 0.6 0.4])
add_probabilities!(diagram, "C", [0.7 0.3; 0.3 0.7])
add_utilities!(diagram, "V", [1.5, 1.7])

using JuMP
model, z, variables = generate_model(diagram, model_type="RJT")

using HiGHS
optimizer = optimizer_with_attributes(
    () -> HiGHS.Optimizer()
)
set_optimizer(model, optimizer)
optimize!(model)

Z = DecisionStrategy(diagram, z)