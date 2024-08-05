using Test, Logging, Random
using DecisionProgramming
using DataStructures

@info "Testing ChanceNode"
@test isa(ChanceNode("A", [], ["a", "b"]), ChanceNode)
@test isa(ChanceNode("B", [], ["x", "y", "z"]), ChanceNode)
@test_throws MethodError ChanceNode(1, [], ["x", "y", "z"])
@test_throws MethodError ChanceNode("B", [1], ["y", "z"])
@test_throws MethodError ChanceNode("B", ["A"], [1, "y", "z"])
@test_throws MethodError ChanceNode("B", ["A"])

@info "Testing DecisionNode"
@test isa(DecisionNode("D", [], ["x", "y"]), DecisionNode)
@test isa(DecisionNode("E", ["C"], ["x", "y"]), DecisionNode)
@test_throws MethodError DecisionNode(1, [], ["x", "y", "z"])
@test_throws MethodError DecisionNode("D", [1], ["y", "z"])
@test_throws MethodError DecisionNode("D", ["A"], [1, "y", "z"])
@test_throws MethodError DecisionNode("D", ["A"])

@info "Testing ValueNode"
@test isa(ValueNode("V", []), ValueNode)
@test isa(ValueNode("V", ["E", "D"]), ValueNode)
@test_throws MethodError ValueNode(1, [])
@test_throws MethodError ValueNode("V", [2])

@info "Testing State"
@test isa(States(State[1, 2, 3]), States)
@test_throws DomainError States(State[0, 1])

@info "Testing paths"
@test vec(collect(paths(States(State[2, 3])))) == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]
@test vec(collect(paths(States(State[2, 3]), Dict(Node(1)=>State(2))))) == [(2, 1), (2, 2), (2, 3)]
@test vec(collect(paths(States(State[2, 3]), FixedPath(Dict(Node(1)=>State(2)))))) == [(2, 1), (2, 2), (2, 3)]

@info "Testing Probabilities"
@test isa(Probabilities([0.4 0.6; 0.3 0.7]), Probabilities)
@test isa(Probabilities([0.0, 0.4, 0.6]), Probabilities)
@test_throws DomainError Probabilities([1.1, 0.1])

@info "Testing DefaultPathProbability"
P = DefaultPathProbability(
    [Node(1), Node(2)],
    [Node[], [Node(1)]],
    [Probabilities([0.4, 0.6]), Probabilities([0.3 0.7; 0.9 0.1])]
)
@test isa(P, DefaultPathProbability)
@test P((State(1), State(2))) == 0.4 * 0.7

@info "Testing Utilities"
@test isa(Utilities(Utility[-1.1, 0.0, 2.7]), Utilities)
@test isa(Utilities(Utility[-1.1 0.0; 2.7 7.0]), Utilities)

@info "Testing DefaultPathUtility"
U = DefaultPathUtility(
    [Node[2], Node[1, 2]],
    [Utilities(Utility[1.0, 1.4]), Utilities(Utility[1.0 1.5; 0.6 3.4])]
)
@test isa(U, DefaultPathUtility)
@test U((State(2), State(1))) == Utility(1.0 + 0.6)

@info "Testing InfluenceDiagram"
diagram = InfluenceDiagram()
@test isa(diagram, InfluenceDiagram)
diagram.Nodes["A"] = ChanceNode("A", [], ["a", "b"])
@test isa(diagram, InfluenceDiagram)

@info "Testing add_node! and validate_node"
diagram = InfluenceDiagram()
add_node!(diagram, ChanceNode("A", [], ["a", "b"]))
@test isa(diagram.Nodes["A"], ChanceNode)
@test_throws DomainError add_node!(diagram, ChanceNode("A", [], ["a", "b"]))
@test_throws DomainError add_node!(diagram, ChanceNode("C", ["B", "B"], ["a", "b"]))
@test_throws DomainError add_node!(diagram, ChanceNode("C", ["C", "B"], ["a", "b"]))
@test_throws DomainError add_node!(diagram, DecisionNode("A", [], ["a", "b"]))
@test_throws DomainError add_node!(diagram, DecisionNode("C", ["B", "B"], ["a", "b"]))
@test_throws DomainError add_node!(diagram, DecisionNode("C", ["C", "B"], ["a", "b"]))
@test_throws DomainError add_node!(diagram, ValueNode("A", []))
@test_throws DomainError add_node!(diagram, ValueNode("C", ["B", "B"]))
@test_throws DomainError add_node!(diagram, ValueNode("C", ["C", "B"]))
add_node!(diagram, ChanceNode("C", ["A"], ["a", "b"]))
add_node!(diagram, DecisionNode("D", ["A"], ["c", "d"]))
add_node!(diagram, ValueNode("V", ["A"]))
@test length(diagram.Nodes) == 4
@test isa(diagram.Nodes["D"], DecisionNode)
@test isa(diagram.Nodes["V"], ValueNode)

@info "Testing generate_arcs!"
diagram = InfluenceDiagram()
@test_throws DomainError generate_arcs!(diagram)
add_node!(diagram, ChanceNode("A", [], ["a", "b"]))
add_node!(diagram, ChanceNode("C", ["A"], ["a", "b", "c"]))
add_node!(diagram, ValueNode("V", ["A", "C"]))
generate_arcs!(diagram)

@test diagram.Names == ["A", "C", "V"]
@test diagram.I_j == OrderedDict("A" => [], "C" => ["A"], "V" => ["A", "C"])
@test diagram.States == OrderedDict("A" => ["a", "b"], "C" => ["a", "b", "c"])
@test diagram.S == OrderedDict{String, Int16}("A" => 2, "C" => 3)
@test string(diagram.C) == string(OrderedDict{String, ChanceNode}("A" => ChanceNode("A", String[], ["a", "b"], 1), "C" => ChanceNode("C", ["A"], ["a", "b", "c"], 2)))
@test diagram.D == OrderedDict{String, DecisionNode}()
@test string(diagram.V) == string(OrderedDict{String, ValueNode}("V" => ValueNode("V", ["A", "C"], 3)))
@test diagram.X == OrderedDict{String, Probabilities}()
@test diagram.Y == OrderedDict{String, Utilities}()

#Non-existent node B
diagram = InfluenceDiagram()
add_node!(diagram, ChanceNode("A", [], ["a", "b"]))
add_node!(diagram, ChanceNode("C", ["B"], ["a", "b", "c"]))
add_node!(diagram, ValueNode("V", ["A", "C"]))
@test_throws DomainError generate_arcs!(diagram)

#Cyclic
diagram = InfluenceDiagram()
add_node!(diagram, ChanceNode("A", ["R"], ["a", "b"]))
add_node!(diagram, ChanceNode("R", ["C", "A"], ["a", "b", "c"]))
add_node!(diagram, ChanceNode("C", ["A"], ["a", "b", "c"]))
add_node!(diagram, ValueNode("V", ["A", "C"]))
@test_throws DomainError generate_arcs!(diagram)

#Value node in I_j
diagram = InfluenceDiagram()
add_node!(diagram, ChanceNode("A", ["R"], ["a", "b"]))
add_node!(diagram, ChanceNode("R", ["C", "A"], ["a", "b", "c"]))
add_node!(diagram, ChanceNode("C", ["A", "V"], ["a", "b", "c"]))
add_node!(diagram, ValueNode("V", ["A"]))
@test_throws DomainError generate_arcs!(diagram)

@info "Testing ProbabilityMatrix"
diagram = InfluenceDiagram()
add_node!(diagram, ChanceNode("A", [], ["a", "b"]))
add_node!(diagram, ChanceNode("B", ["A"], ["a", "b", "c"]))
add_node!(diagram, DecisionNode("D", ["A"], ["a", "b", "c"]))
generate_arcs!(diagram)
@test ProbabilityMatrix(diagram, "A") == zeros(2)
@test ProbabilityMatrix(diagram, "B") == zeros(2, 3)
@test_throws DomainError ProbabilityMatrix(diagram, "C")
@test_throws DomainError ProbabilityMatrix(diagram, "D")
X_A = ProbabilityMatrix(diagram, "A")
X_A["a"] = 0.2
@test X_A  == [0.2, 0]
X_A["b"] = 0.9
@test X_A  == [0.2, 0.9]
@test_throws DomainError add_probabilities!(diagram, "A", X_A)
X_A["b"] = 0.8
@test add_probabilities!(diagram, "A", X_A) == [0.2, 0.8]
@test_throws DomainError add_probabilities!(diagram, "A", X_A)

@info "Testing UtilityMatrix"
diagram = InfluenceDiagram()
add_node!(diagram, ChanceNode("A", [], ["a", "b"]))
add_node!(diagram, DecisionNode("D", ["A"], ["a", "b", "c"]))
add_node!(diagram, ValueNode("V", ["A", "D"]))
generate_arcs!(diagram)
@test UtilityMatrix(diagram, "V") == fill(Inf, (2, 3))
@test_throws DomainError UtilityMatrix(diagram, "C")
@test_throws DomainError UtilityMatrix(diagram, "D")
Y_V = UtilityMatrix(diagram, "V")
@test_throws DomainError add_utilities!(diagram, "V", Y_V)
Y_V["a", :] = [1, 2, 3]
Y_V["b", "c"] = 4
Y_V["b", "a"] = 5
Y_V["b", "b"] = 6
@test Y_V  == [1 2 3; 5 6 4]
add_utilities!(diagram, "V", Y_V)

@test diagram.Y["V"].data == [1.0 2.0 3.0; 5.0 6.0 4.0]
@test_throws DomainError add_utilities!(diagram, "V", Y_V)

@info "Testing generate_diagram!"
diagram = InfluenceDiagram()
add_node!(diagram, ChanceNode("A", [], ["a", "b"]))
add_node!(diagram, DecisionNode("D", ["A"], ["a", "b", "c"]))
add_node!(diagram, ValueNode("V", ["A", "D"]))
generate_arcs!(diagram)
add_utilities!(diagram, "V", [-1 2 3; 5 6 4])
add_probabilities!(diagram, "A", [0.2, 0.8])
generate_diagram!(diagram)
@test diagram.translation == Utility(0)

@info "Testing positive and negative path utility translations"
generate_diagram!(diagram, positive_path_utility=true)
@test diagram.translation == Utility(2)
@test all(diagram.U(s, diagram.translation) > 0 for s in paths(get_values(diagram.S)))
generate_diagram!(diagram, negative_path_utility=true)
@test diagram.translation == Utility(-7)
@test all(diagram.U(s, diagram.translation) < 0 for s in paths(get_values(diagram.S)))
