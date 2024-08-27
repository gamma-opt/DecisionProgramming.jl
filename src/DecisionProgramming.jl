module DecisionProgramming

include("influence_diagram.jl")
include("decision_model.jl")
include("analysis.jl")
include("heuristics.jl")
include("printing.jl")

export Node,
    Name,
    AbstractNode,
    ChanceNode,
    DecisionNode,
    ValueNode,
    State,
    States,
    Path,
    paths,
    ForbiddenPath,
    FixedPath,
    Probabilities,
    Utility,
    Utilities,
    AbstractPathProbability,
    DefaultPathProbability,
    AbstractPathUtility,
    DefaultPathUtility,
    validate_structure,
    RJT,
    InfluenceDiagram,
    generate_arcs!,
    generate_diagram!,
    indices,
    I_j_indices,
    indices_in_vector,
    get_values,
    get_keys,
    num_states,
    add_node!,
    ProbabilityMatrix,
    add_probabilities!,
    UtilityMatrix,
    add_utilities!,
    LocalDecisionStrategy,
    DecisionStrategy

export DecisionVariables,
    PathCompatibilityVariables,
    lazy_probability_cut,
    expected_value,
    conditional_value_at_risk,
    RJT_conditional_value_at_risk,
    ID_to_RJT,
    RJTVariables,
    generate_model

export random_diagram!,
    random_probabilities!,
    random_utilities!,
    LocalDecisionStrategy

export CompatiblePaths,
    UtilityDistribution,
    StateProbabilities,
    value_at_risk,
    conditional_value_at_risk

export randomStrategy,
    singlePolicyUpdate

export print_decision_strategy,
    print_utility_distribution,
    print_state_probabilities,
    print_statistics,
    print_risk_measures,
    print_node_io,
    print_node,
    print_diagram,
    nodes,
    graph,
    mermaid

# For API docs
export AbstractRNG, Model, VariableRef

end # module
