using Test, Logging, Random, JuMP
using DecisionProgramming


function influence_diagram()
    diagram = InfluenceDiagram()
    #Creating an experimental influence diagram with nodes having varying amounts of states and nodes in their information sets
    add_node!(diagram, ChanceNode("H1", [], ["1", "2"]))
    add_node!(diagram, DecisionNode("D1", [], ["1", "2"]))
    add_node!(diagram, DecisionNode("D2", ["H1"], ["1", "2", "3"]))
    add_node!(diagram, ChanceNode("H2", ["D1", "D2"], ["1", "2", "3", "4"]))
    add_node!(diagram, ChanceNode("H3", [], ["1", "2", "3"]))
    add_node!(diagram, DecisionNode("D3", ["H2", "H3"], ["1", "2"]))

    add_node!(diagram, ValueNode("C1", ["D2", "H2"]))
    add_node!(diagram, ValueNode("C2", ["D3"]))

    generate_arcs!(diagram)

    H1_probs = 1/length(diagram.Nodes["H1"].states)
    add_probabilities!(diagram, "H1", [H1_probs, H1_probs])

    H2_probs = 1/length(diagram.Nodes["H2"].states)
    H2_prob_matrix_values = [H2_probs, H2_probs, H2_probs, H2_probs]
    H2_prob_matrix = ProbabilityMatrix(diagram, "H2")
    for state1 in ["1", "2"]
        for state2 in ["1", "2", "3"]
            setindex!(H2_prob_matrix, H2_prob_matrix_values, state1, state2, Colon())
        end
    end
    add_probabilities!(diagram, "H2", H2_prob_matrix)

    H3_probs = 1/length(diagram.Nodes["H3"].states)
    add_probabilities!(diagram, "H3", [H3_probs, H3_probs, H3_probs])

    C1_util_matrix = UtilityMatrix(diagram, "C1")
    C1_util_matrix["1", :] = [0.0, 0.0, 0.0, 0.0]
    C1_util_matrix["2", :] = [0.0, 0.0, 0.0, 0.0]
    C1_util_matrix["3", :] = [0.0, 0.0, 0.0, 0.0]
    add_utilities!(diagram, "C1", C1_util_matrix)

    add_utilities!(diagram, "C2", [0.0, 0.0])

    generate_diagram!(diagram, positive_path_utility = true)

    return diagram
end

function test_decision_model(diagram, probability_scale_factor, probability_cut)
    model = Model()

    @info "Testing DecisionVariables"
    z = DecisionVariables(model, diagram)

    @info "Testing PathCompatibilityVariables"
    if probability_scale_factor > 0
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
    Z_j = [LocalDecisionStrategy(rng, diagram, d) for d in keys(diagram.D)]
    D_indices = indices(diagram.D)
    D_I_j_indices = I_j_indices(diagram, diagram.D)
    Z = DecisionStrategy(D_indices, D_I_j_indices, Z_j)

    @info "Testing CompatiblePaths"
    @test all(true for s in CompatiblePaths(diagram, Z))
    @test_throws DomainError CompatiblePaths(diagram, Z, Dict(diagram.Nodes["D1"].index => State(1)))
    node, state = (diagram.Nodes["H1"].index, State(1))
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
    print_state_probabilities(diagram, S_probabilities, get_keys(diagram.C))
    print_state_probabilities(diagram, S_probabilities, get_keys(diagram.D))
    print_state_probabilities(diagram, S_probabilities2, get_keys(diagram.C))
    print_state_probabilities(diagram, S_probabilities2, get_keys(diagram.D))
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


