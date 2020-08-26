using Logging, Test, Random
using DecisionProgramming

rng = MersenneTwister(4)

@info "Testing random_diagram"
@test_throws DomainError random_diagram(rng, -1, 1, 1, 1, 1)
@test_throws DomainError random_diagram(rng, 1, -1, 1, 1, 1)
@test_throws DomainError random_diagram(rng, 0, 0, 1, 1, 1)
@test_throws DomainError random_diagram(rng, 1, 1, 0, 1, 1)
@test_throws DomainError random_diagram(rng, 1, 1, 1, 0, 1)
@test_throws DomainError random_diagram(rng, 1, 1, 1, 1, 0)

for (n_C, n_D) in [(1, 0), (0, 1)]
    rng = RandomDevice()
    C, D, V = random_diagram(rng, n_C, n_D, 1, 1, 1)
    @test isa(C, Vector{ChanceNode})
    @test isa(D, Vector{DecisionNode})
    @test isa(V, Vector{ValueNode})
    @test length(C) == n_C
    @test length(D) == n_D
    @test length(V) == 1
    @test all(!isempty(v.I_j) for v in V)
end

@info "Testing random States"
@test_throws DomainError States(rng, [0], 10)
@test isa(States(rng, [2, 3], 10), States)

@info "Testing random Probabilities"
S = States([2, 3, 2])
c = ChanceNode(3, [1, 2])
@test isa(Probabilities(rng, c, S; n_inactive=0), Probabilities)
@test isa(Probabilities(rng, c, S; n_inactive=1), Probabilities)
@test isa(Probabilities(rng, c, S; n_inactive=2*3*(2-1)), Probabilities)
@test_throws DomainError Probabilities(rng, c, S; n_inactive=2*3*(2-1)+1)

@info "Testing random Consequences"
S = States([2, 3])
v = ValueNode(3, [1, 2])
@test isa(Consequences(rng, v, S; low=-1.0, high=1.0), Consequences)
@test_throws DomainError Consequences(rng, v, S; low=1.1, high=1.0)

@info "Testing random LocalDecisionStrategy"
S = States([2, 3, 2])
d = DecisionNode(3, [1, 2])
@test isa(LocalDecisionStrategy(rng, d, S), LocalDecisionStrategy)
