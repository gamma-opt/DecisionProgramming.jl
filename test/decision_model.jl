using Test, Logging, Random, JuMP
using DecisionProgramming


function influence_diagram(rng::AbstractRNG, n_C::Int, n_D::Int, n_V::Int, m_C::Int, m_D::Int, states::Vector{Int}, n_inactive::Int)
    C, D, V = random_diagram(rng, n_C, n_D, n_V, m_C, m_D)
    S = States(rng, states, length(C) + length(D))
    X = [Probabilities(rng, c, S; n_inactive=n_inactive) for c in C]
    Y = [Consequences(rng, v, S; low=-1.0, high=1.0) for v in V]

    validate_influence_diagram(S, C, D, V)

    s_c = sortperm([c.j for c in C])
    s_d = sortperm([d.j for d in D])
    s_v = sortperm([v.j for v in V])
    C, D, V = C[s_c], D[s_d], V[s_v]
    X, Y = X[s_c], Y[s_v]
    P = DefaultPathProbability(C, X)
    U = DefaultPathUtility(V, Y)

    return D, S, P, U
end

function test_decision_model(D, S, P, U, n_inactive, hard_lower_bound)
    model = Model()

    @info "Testing decision_variables"
    z = decision_variables(model, S, D)

    @info "Testing path_probability_variables"
    π_s = path_probability_variables(model, z, S, D, P; hard_lower_bound = hard_lower_bound)

    @info "Testing PositivePathUtility"
    U′ = if hard_lower_bound U else PositivePathUtility(S, U) end

    @info "Testing probability_cut"
    probability_cut(model, π_s, P)

    @info "Testing active_paths_cut"
    if iszero(n_inactive)
        active_paths_cut(model, π_s, S, P)
    else
        @test_throws DomainError active_paths_cut(model, π_s, S, P)
    end

    @info "Testing expected_value"
    EV = expected_value(model, π_s, U′)

    @info "Testing conditional_value_at_risk"
    CVaR = conditional_value_at_risk(model, π_s, U′, 0.2)

    @test true
end

function test_analysis_and_printing(D, S, P, U)
    @info("Creating random decision strategy")
    Z_j = [LocalDecisionStrategy(rng, d, S) for d in D]
    Z = DecisionStrategy(D, Z_j)

    @info "Testing CompatiblePaths"
    @test all(true for s in CompatiblePaths(S, P.C, Z))
    @test_throws DomainError CompatiblePaths(S, P.C, Z, Dict(D[1].j => 1))
    node, state = (P.C[1].j, 1)
    @test all(s[node] == state for s in CompatiblePaths(S, P.C, Z, Dict(node => state)))

    @info "Testing UtilityDistribution"
    udist = UtilityDistribution(S, P, U, Z)

    @info "Testing StateProbabilities"
    sprobs = StateProbabilities(S, P, Z)

    @info "Testing conditional StateProbabilities"
    sprobs2 = StateProbabilities(S, P, Z, node, state, sprobs)

    @info "Testing "
    print_decision_strategy(S, Z)
    print_utility_distribution(udist)
    print_state_probabilities(sprobs, [c.j for c in P.C])
    print_state_probabilities(sprobs, [d.j for d in D])
    print_state_probabilities(sprobs2, [c.j for c in P.C])
    print_state_probabilities(sprobs2, [d.j for d in D])
    print_statistics(udist)
    print_risk_measures(udist, [0.0, 0.05, 0.1, 0.2, 1.0])

    @test true
end

@info "Testing model construction"
rng = MersenneTwister(4)
for (n_C, n_D, states, n_inactive, hard_lower_bound) in [
        (3, 2, [1, 2, 3], 0, true),
        (3, 2, [3], 1, true),
        (3, 2, [1, 2, 3], 0, false),
        (3, 2, [3], 1, false)
    ]
    D, S, P, U = influence_diagram(rng, n_C, n_D, 2, 2, 2, states, n_inactive)
    test_decision_model(D, S, P, U, n_inactive, hard_lower_bound)
    test_analysis_and_printing(D, S, P, U)
end
