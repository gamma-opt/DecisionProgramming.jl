using DataFrames, PrettyTables
using StatsBase, StatsBase.Statistics

"""
    function print_decision_strategy(S::States, Z::DecisionStrategy)

Print decision strategy.

# Examples
```julia
print_decision_strategy(S, Z)
```
"""
function print_decision_strategy(diagram::InfluenceDiagram, Z::DecisionStrategy, state_probabilities::StateProbabilities; show_incompatible_states = false)
    probs = state_probabilities.probs

    for (d, I_d, Z_d) in zip(Z.D, Z.I_d, Z.Z_d)
        s_I = vec(collect(paths(diagram.S[I_d])))
        s_d = [Z_d(s) for s in s_I]

        if !isempty(I_d)
            informations_states = [join([String(diagram.States[i][s_i]) for (i, s_i) in zip(I_d, s)], ", ") for s in s_I]
            decision_probs = [ceil(prod(probs[i][s1] for (i, s1) in zip(I_d, s))) for s in s_I]
            decisions = collect(p == 0 ? "--" : diagram.States[d][s] for (s, p) in zip(s_d, decision_probs))
            df = DataFrame(informations_states = informations_states, decisions = decisions)
            if !show_incompatible_states
                 filter!(row -> row.decisions != "--", df)
            end
            pretty_table(df, ["State(s) of $(join([diagram.Names[i] for i in I_d], ", "))", "Decision in $(diagram.Names[d])"], alignment=:l)
        else
            df = DataFrame(decisions = diagram.States[d][s_d])
            pretty_table(df, ["Decision in $(diagram.Names[d])"], alignment=:l)
        end
    end
end

"""
    function print_utility_distribution(udist::UtilityDistribution; util_fmt="%f", prob_fmt="%f")

Print utility distribution.

# Examples
```julia
udist = UtilityDistribution(S, P, U, Z)
print_utility_distribution(udist)
```
"""
function print_utility_distribution(udist::UtilityDistribution; util_fmt="%f", prob_fmt="%f")
    df = DataFrame(Utility = udist.u, Probability = udist.p)
    formatters = (
        ft_printf(util_fmt, [1]),
        ft_printf(prob_fmt, [2]))
    pretty_table(df; formatters = formatters)
end

"""
    function print_state_probabilities(sprobs::StateProbabilities, nodes::Vector{Node}; prob_fmt="%f")

Print state probabilities with fixed states.

# Examples
```julia
sprobs = StateProbabilities(S, P, U, Z)
print_state_probabilities(sprobs, [c.j for c in C])
print_state_probabilities(sprobs, [d.j for d in D])
```
"""
function print_state_probabilities(diagram::InfluenceDiagram, state_probabilities::StateProbabilities, nodes::Vector{Name}; prob_fmt="%f")
    node_indices = [findfirst(j -> j ==node, diagram.Names) for node in nodes]

    probs = state_probabilities.probs
    fixed = state_probabilities.fixed

    prob(p, state) = if 1≤state≤length(p) p[state] else NaN end
    fix_state(i) = if i∈keys(fixed) string(fixed[i]) else "" end

    # Maximum number of states
    limit = maximum(length(probs[i]) for i in node_indices)
    states = 1:limit
    df = DataFrame()
    df[!, :Node] = nodes
    for state in states
        df[!, Symbol("State $state")] = [prob(probs[i], state) for i in node_indices]
    end
    df[!, Symbol("Fixed state")] = [fix_state(i) for i in node_indices]
    pretty_table(df; formatters = ft_printf(prob_fmt, (first(states)+1):(last(states)+1)))
end

"""
function print_statistics(udist::UtilityDistribution; fmt = "%f")

Print statistics about utility distribution.
"""
function print_statistics(udist::UtilityDistribution; fmt = "%f")
    u = udist.u
    w = ProbabilityWeights(udist.p)
    names = ["Mean", "Std", "Skewness", "Kurtosis"]
    statistics = [mean(u, w), std(u, w, corrected=false), skewness(u, w), kurtosis(u, w)]
    df = DataFrame(Name = names, Statistics = statistics)
    pretty_table(df, formatters = ft_printf(fmt, [2]))
end

"""
    function print_risk_measures(udist::UtilityDistribution, αs::Vector{Float64}; fmt = "%f")

Print risk measures.
"""
function print_risk_measures(udist::UtilityDistribution, αs::Vector{Float64}; fmt = "%f")
    u, p = udist.u, udist.p
    VaR = [value_at_risk(u, p, α) for α in αs]
    CVaR = [conditional_value_at_risk(u, p, α) for α in αs]
    df = DataFrame(α = αs, VaR = VaR, CVaR = CVaR)
    pretty_table(df, formatters = ft_printf(fmt))
end
