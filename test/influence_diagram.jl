using Test, Logging, Random
using DecisionProgramming

@info "Testing ChandeNode"
@test isa(ChanceNode(1, Node[]), ChanceNode)
@test isa(ChanceNode(2, Node[1]), ChanceNode)
@test_throws DomainError ChanceNode(1, Node[1])
@test_throws DomainError ChanceNode(1, Node[2])

@info "Testing DecisionNode"
@test isa(DecisionNode(1, Node[]), DecisionNode)
@test isa(DecisionNode(2, Node[1]), DecisionNode)
@test_throws DomainError DecisionNode(1, Node[1])
@test_throws DomainError DecisionNode(1, Node[2])

@info "Testing ValueNode"
@test isa(ValueNode(1, Node[]), ValueNode)
@test isa(ValueNode(2, Node[1]), ValueNode)
@test_throws DomainError ValueNode(1, Node[1])
@test_throws DomainError ValueNode(1, Node[2])

@info "Testing State"
@test isa(States([1, 2, 3]), States)
@test_throws DomainError States([0, 1])
@test States([(2, [1, 3]), (3, [2, 4, 5])]) == States([2, 3, 2, 3, 3])

@info "Testing validate_influence_diagram"
@test_throws DomainError validate_influence_diagram(
    States([1]),
    [ChanceNode(1, Node[])],
    [DecisionNode(2, Node[])],
    [ValueNode(3, [1, 2])]
)
@test_throws DomainError validate_influence_diagram(
    States([1, 1]),
    [ChanceNode(1, Node[])],
    [DecisionNode(1, Node[])],
    [ValueNode(3, [1, 2])]
)
@test_throws DomainError validate_influence_diagram(
    States([1, 1]),
    [ChanceNode(1, Node[])],
    [DecisionNode(2, Node[])],
    [ValueNode(2, [1])]
)
@test_throws DomainError validate_influence_diagram(
    States([1, 1]),
    [ChanceNode(1, Node[])],
    [DecisionNode(2, Node[])],
    [ValueNode(3, [2]), ValueNode(4, [3])]
)
# Test redundancy
@test validate_influence_diagram(
    States([1, 1]),
    [ChanceNode(1, Node[])],
    [DecisionNode(2, Node[])],
    [ValueNode(3, Node[])]
) === nothing

@info "Testing paths"
@test vec(collect(paths(States([2, 3])))) == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]
@test vec(collect(paths(States([2, 3]), Dict(1=>2)))) == [(2, 1), (2, 2), (2, 3)]

@info "Testing Probabilities"
@test isa(Probabilities(1, [0.4 0.6; 0.3 0.7]), Probabilities)
@test isa(Probabilities(1, [0.0, 0.4, 0.6]), Probabilities)
@test_throws DomainError Probabilities(1, [1.1, 0.1])

@info "Testing DefaultPathProbability"
P = DefaultPathProbability(
    [ChanceNode(1, Node[]), ChanceNode(2, [1])],
    [Probabilities(1, [0.4, 0.6]), Probabilities(2, [0.3 0.7; 0.9 0.1])]
)
@test isa(P, DefaultPathProbability)
@test P((1, 2)) == 0.4 * 0.7

@info "Testing Consequences"
@test isa(Consequences(1, [-1.1, 0.0, 2.7]), Consequences)
@test isa(Consequences(1, [-1.1 0.0; 2.7 7.0]), Consequences)

@info "Testing DefaultPathUtility"
U = DefaultPathUtility(
    [ValueNode(3, [2]), ValueNode(4, [1, 2])],
    [Consequences(3, [1.0, 1.4]), Consequences(4, [1.0 1.5; 0.6 3.4])]
)
@test isa(U, DefaultPathUtility)
@test U((2, 1)) == 1.0 + 0.6

@info "Testing LocalDecisionStrategy"
@test_throws DomainError LocalDecisionStrategy(1, [0, 0, 2])
@test_throws DomainError LocalDecisionStrategy(1, [0, 1, 1])
