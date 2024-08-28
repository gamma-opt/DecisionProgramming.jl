using DecisionProgramming

diagram = InfluenceDiagram()
add_node!(diagram, DecisionNode("D1", [], ["a", "b"]))
add_node!(diagram, ChanceNode("C2", ["D1", "C1"], ["v", "w"]))
add_node!(diagram, ChanceNode("C1", [], ["x", "y", "z"]))

add_node!(diagram, ValueNode("V", ["C2"]))

generate_arcs!(diagram)

X_C2 = zeros(2, 3, 2)
X_C2[1, 1, 1] = 0.2
X_C2[1, 1, 2] = 0.8
X_C2[2, 1, 1] = 0.2
X_C2[2, 1, 2] = 0.8
X_C2[1, 2, 1] = 0.2
X_C2[1, 2, 2] = 0.8
X_C2[2, 2, 1] = 0.2
X_C2[2, 2, 2] = 0.8
X_C2[1, 3, 1] = 0.2
X_C2[1, 3, 2] = 0.8
X_C2[2, 3, 1] = 0.2
X_C2[2, 3, 2] = 0.8