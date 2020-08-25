using Logging, Test, Random, JuMP
using DecisionProgramming

@test isa(ChanceNode(1, Node[]), ChanceNode)
@test isa(ChanceNode(2, Node[1]), ChanceNode)
@test_throws DomainError ChanceNode(1, Node[2])

@test isa(DecisionNode(1, Node[]), DecisionNode)
@test isa(DecisionNode(2, Node[1]), DecisionNode)
@test_throws DomainError DecisionNode(1, Node[2])

@test isa(ValueNode(1, Node[]), ValueNode)
@test isa(ValueNode(2, Node[1]), ValueNode)
@test_throws DomainError ValueNode(1, Node[2])

@test isa(States([1, 2, 3]), States)
@test_throws DomainError States([0, 1])
@test States([(2, [1, 3]), (3, [2, 4, 5])]) == States([2, 3, 2, 3, 3])

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
@test isnothing(validate_influence_diagram(
    States([1, 1]),
    [ChanceNode(1, Node[])],
    [DecisionNode(2, Node[])],
    [ValueNode(3, [1])]
))

@test vec(collect(paths(States([2, 3])))) == [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]
@test vec(collect(paths(States([2, 3]), Dict(1=>2)))) == [(2, 1), (2, 2), (2, 3)]

@test isa(Probabilities([0.4 0.6; 0.3 0.7]), Probabilities)
@test isa(Probabilities([0.0, 0.4, 0.6]), Probabilities)
@test_throws DomainError Probabilities([1.1, 0.1])

P = DefaultPathProbability(
    [ChanceNode(1, Node[]), ChanceNode(2, [1])],
    [Probabilities([0.4, 0.6]), Probabilities([0.3 0.7; 0.9 0.1])]
)
@test isa(P, DefaultPathProbability)
@test P((1, 2)) == 0.4 * 0.7

@test isa(Consequences([-1.1, 0.0, 2.7]), Consequences)
@test isa(Consequences([-1.1 0.0; 2.7 7.0]), Consequences)

U = DefaultPathUtility(
    [ValueNode(3, [2]), ValueNode(4, [1, 2])],
    [Consequences([1.0, 1.4]), Consequences([1.0 1.5; 0.6 3.4])]
)
@test isa(U, DefaultPathUtility)
@test U((2, 1)) == 1.0 + 0.6
