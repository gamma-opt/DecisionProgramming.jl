using Test, Logging, Random, JuMP
using DecisionProgramming


function influence_diagram(rng::AbstractRNG, n_C::Int, n_D::Int, n_V::Int, m_C::Int, m_D::Int, states::Vector{Int}, n_inactive::Int)
    diagram = InfluenceDiagram()
    random_diagram!(rng, diagram, n_C, n_D, n_V, m_C, m_D, states)
    for c in diagram.C
        random_probabilities!(rng, diagram, c; n_inactive=n_inactive)
    end
    for v in diagram.V
        random_utilities!(rng, diagram, v; low=-1.0, high=1.0)
    end

    # Names needed for printing functions only
    diagram.Names = ["node$j" for j in 1:n_C+n_D+n_V]
    diagram.States = [["s$s" for s in 1:n_s] for n_s in diagram.S]

    diagram.P = DefaultPathProbability(diagram.C, diagram.I_j[diagram.C], diagram.X)
    diagram.U = DefaultPathUtility(diagram.I_j[diagram.V], diagram.Y)

    return diagram
end

function test_decision_model(diagram, n_inactive, probability_scale_factor, probability_cut)
    model = Model()

    @info "Testing DecisionVariables"
    z = DecisionVariables(model, diagram)

    @info "Testing PathCompatibilityVariables"
    x_s = PathCompatibilityVariables(model, diagram, z; probability_cut = probability_cut)

    @info "Testing probability_cut"
    lazy_probability_cut(model, diagram, x_s)

    @info "Testing expected_value"
    if probability_scale_factor > 0
        EV = expected_value(model, diagram, x_s; probability_scale_factor = probability_scale_factor)
    else
        @test_throws DomainError expected_value(model, diagram, x_s; probability_scale_factor = probability_scale_factor)
    end

    @info "Testing conditional_value_at_risk"
    if probability_scale_factor > 0
        CVaR = conditional_value_at_risk(model, diagram, x_s, 0.2; probability_scale_factor = probability_scale_factor)
    else
        @test_throws DomainError conditional_value_at_risk(model, diagram, x_s, 0.2; probability_scale_factor = probability_scale_factor)
    end

    @test true
end

function test_analysis_and_printing(diagram)
    @info("Creating random decision strategy")
    Z_j = [LocalDecisionStrategy(rng, diagram, d) for d in diagram.D]
    Z = DecisionStrategy(diagram.D, diagram.I_j[diagram.D], Z_j)

    @info "Testing CompatiblePaths"
    @test all(true for s in CompatiblePaths(diagram, Z))
    @test_throws DomainError CompatiblePaths(diagram, Z, Dict(diagram.D[1] => State(1)))
    node, state = (diagram.C[1], State(1))
    @test all(s[node] == state for s in CompatiblePaths(diagram, Z, Dict(node => state)))

    @info "Testing UtilityDistribution"
    U_distribution = UtilityDistribution(diagram, Z)

    @info "Testing StateProbabilities"
    S_probabilities = StateProbabilities(diagram, Z)

    @info "Testing conditional StateProbabilities"
    S_probabilities2 = StateProbabilities(diagram, Z, node, state, S_probabilities)

    @info "Testing "
    print_decision_strategy(diagram, Z, S_probabilities)
    print_decision_strategy(diagram, Z, S_probabilities, show_incompatible_states=true)
    print_utility_distribution(U_distribution)
    print_state_probabilities(diagram, S_probabilities, [diagram.Names[c] for c in diagram.C])
    print_state_probabilities(diagram, S_probabilities, [diagram.Names[d] for d in diagram.D])
    print_state_probabilities(diagram, S_probabilities2, [diagram.Names[c] for c in diagram.C])
    print_state_probabilities(diagram, S_probabilities2, [diagram.Names[d] for d in diagram.D])
    print_statistics(U_distribution)
    print_risk_measures(U_distribution, [0.0, 0.05, 0.1, 0.2, 1.0])

    @test true
end

@info "Testing model construction"
rng = MersenneTwister(4)
for (n_C, n_D, states, n_inactive, probability_scale_factor, probability_cut) in [
        (3, 2, [2, 3, 4], 0, 1.0, true),
        (3, 2, [2, 3], 0, -1.0, true),
        (3, 2, [3], 1, 100.0, true),
        (3, 2, [2, 3], 0, -1.0, false),
        (3, 2, [4], 1, 10.0, false)
    ]
    diagram = influence_diagram(rng, n_C, n_D, 2, 2, 2, states, n_inactive)
    test_decision_model(diagram, n_inactive, probability_scale_factor, probability_cut)
    test_analysis_and_printing(diagram)
end
