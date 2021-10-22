using Test, Logging, Random
using DecisionProgramming

rng = MersenneTwister(4)

@info "Testing random_diagram"
diagram = InfluenceDiagram()
@test_throws DomainError random_diagram!(rng, diagram, -1, 1, 1, 1, 1, [2])
@test_throws DomainError random_diagram!(rng, diagram, 1, -1, 1, 1, 1, [2])
@test_throws DomainError random_diagram!(rng, diagram, 0, 0, 1, 1, 1, [2])
@test_throws DomainError random_diagram!(rng, diagram, 1, 1, 0, 1, 1, [2])
@test_throws DomainError random_diagram!(rng, diagram, 1, 1, 1, 0, 1, [2])
@test_throws DomainError random_diagram!(rng, diagram, 1, 1, 1, 1, 0, [2])
@test_throws DomainError random_diagram!(rng, diagram, 1, 1, 1, 1, 1, [1])

for (n_C, n_D) in [(1, 0), (0, 1)]
    rng = RandomDevice()
    diagram = InfluenceDiagram()
    random_diagram!(rng, diagram, n_C, n_D, 1, 1, 1, [2])
    @test isa(diagram.C, Vector{Node})
    @test isa(diagram.D, Vector{Node})
    @test isa(diagram.V, Vector{Node})
    @test length(diagram.C) == n_C
    @test length(diagram.D) == n_D
    @test length(diagram.V) == 1
    @test all(!isempty(I_v) for I_v in diagram.I_j[diagram.V])
    @test isa(diagram.S, States)
end

@info "Testing random Probabilities"
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 2, 2, 1, 1, 1, [2])
@test_throws DomainError random_probabilities!(rng, diagram, diagram.D[1]; n_inactive=0)
random_probabilities!(rng, diagram, diagram.C[1]; n_inactive=0)
@test isa(diagram.X[1], Probabilities)
random_probabilities!(rng, diagram, diagram.C[2]; n_inactive=1)
@test isa(diagram.X[2], Probabilities)

diagram = InfluenceDiagram()
diagram.C = Node[1,3]
diagram.D = Node[2]
diagram.I_j = [Node[], Node[], Node[1,2]]
diagram.S = States(State[2, 3, 2])
diagram.X = Vector{Probabilities}(undef, 2)
random_probabilities!(rng, diagram, diagram.C[2]; n_inactive=2*3*(2-1))
@test isa(diagram.X[2], Probabilities)
@test_throws DomainError random_probabilities!(rng, diagram, diagram.C[2]; n_inactive=2*3*(2-1)+1)


@info "Testing random Utilities"
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 1, 1, 2, 1, 1, [2])
random_utilities!(rng, diagram, diagram.V[1], low=-1.0, high=1.0)
@test isa(diagram.Y[1], Utilities)
@test_throws DomainError random_utilities!(rng, diagram, diagram.V[1], low=1.1, high=1.0)
@test_throws DomainError random_utilities!(rng, diagram, diagram.C[1], low=-1.0, high=1.0)

@info "Testing random LocalDecisionStrategy"
diagram = InfluenceDiagram()
random_diagram!(rng, diagram, 2, 2, 2, 2, 2, [2])
@test isa(LocalDecisionStrategy(rng, diagram, diagram.D[1]), LocalDecisionStrategy)
