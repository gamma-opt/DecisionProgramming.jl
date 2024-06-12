using Test, Logging, Random, JuMP
using DecisionProgramming


function influence_diagram(rng::AbstractRNG, n_C::Int, n_D::Int, n_V::Int, m_C::Int, m_D::Int, states::Vector{Int}, n_inactive::Int)
    diagram = InfluenceDiagram()
    #tähän esimerkki suoraan eri nodeja, eri statien määriä, eri lähtönodejen ja päätösnodejen määriä
    random_diagram!(rng, diagram, n_C, n_D, n_V, m_C, m_D, states)
    for c in keys(diagram.C)
        random_probabilities!(rng, diagram, c; n_inactive=n_inactive)
    end
    for v in keys(diagram.V)
        random_utilities!(rng, diagram, v; low=-1.0, high=1.0)
    end

    # Names needed for printing functions only
    diagram.Names = ["$j" for j in 1:n_C+n_D+n_V]
    println("")
    println(values(diagram.S))
    #diagram.States = [["s$s" for s in 1:n_s] for n_s in collect(values(diagram.S))]

    C_I_j = [get(diagram.I_j, key, Set{Int}()) for key in keys(diagram.C)]
    C_I_j_int16 = [Int16.(parse.(Int, collect(s))) for s in C_I_j]

    V_I_j = [get(diagram.I_j, key, Set{Int}()) for key in keys(diagram.V)]
    V_I_j_int16 = [Int16.(parse.(Int, collect(s))) for s in V_I_j]

    diagram.P = DefaultPathProbability(parse.(Int16, keys(diagram.C)), C_I_j_int16, collect(values(diagram.X)))
    diagram.U = DefaultPathUtility(V_I_j_int16, collect(values(diagram.Y)))

    return diagram
end

function test_decision_model(diagram, n_inactive, probability_scale_factor, probability_cut)
    model = Model()

    @info "Testing DecisionVariables"
    z = DecisionVariables(model, diagram)

    @info "Testing PathCompatibilityVariables"
    if probability_scale_factor > 0
        println("model:")
        println(model)
        println(diagram)
        println(z)
        println(diagram.P)
        x_s = PathCompatibilityVariables(model, diagram, z; probability_cut = probability_cut, probability_scale_factor = probability_scale_factor)
    else
        @test_throws DomainError x_s = PathCompatibilityVariables(model, diagram, z; probability_cut = probability_cut, probability_scale_factor = probability_scale_factor)
    end

    x_s = PathCompatibilityVariables(model, diagram, z; probability_cut = probability_cut, probability_scale_factor = 1.0)    

    @info "Testing probability_cut"
    lazy_probability_cut(model, diagram, x_s)

    @info "Testing expected_value"
    EV = expected_value(model, diagram, x_s)

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
    #laita vain joku decision node tähän
    @test_throws DomainError CompatiblePaths(diagram, Z, Dict(diagram.D[1] => State(1)))
    #joku chance node vain tähän
    node, state = (diagram.C[1], State(1))
    @test all(s[node] == state for s in CompatiblePaths(diagram, Z, Dict(node => state)))

    @info "Testing UtilityDistribution"
    U_distribution = UtilityDistribution(diagram, Z)

    @info "Testing StateProbabilities"
    S_probabilities = StateProbabilities(diagram, Z)

    @info "Testing conditional StateProbabilities"
    S_probabilities2 = StateProbabilities(diagram, Z, node, state, S_probabilities)

    @info "Testing printing functions"
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
for (probability_scale_factor, probability_cut) in [
        (1.0, true),
        (-1.0, true),
        (100.0, true),
        (-1.0, false),
        (10.0, false)
    ]
    diagram = influence_diagram()
    test_decision_model(diagram, probability_scale_factor, probability_cut)
    test_analysis_and_printing(diagram)
end
