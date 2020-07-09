module DecisionProgramming

include("model.jl")
export DecisionModel,
    Specs,
    InfluenceDiagram,
    Params,
    paths,
    path_probability,
    minimum_path_probability,
    path_utility,
    probability_sum_cut,
    number_of_paths_cut

include("analysis.jl")
export active_paths,
    is_compatible,
    state_probabilities,
    utility_distribution,
    print_results,
    print_decision_strategy,
    print_state_probabilities

end # module
