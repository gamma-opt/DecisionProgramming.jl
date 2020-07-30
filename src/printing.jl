using Printf, Parameters, DataFrames, PrettyTables
using StatsBase, StatsBase.Statistics

"""Print decision strategy.

# Examples
```julia
print_decision_strategy(G, Z)
```
"""
function print_decision_strategy(G::InfluenceDiagram, Z::DecisionStrategy)
    @unpack C, D, V, I_j, S_j = G
    for j in D
        a1 = collect(paths(S_j[I_j[j]]))[:]
        a2 = [findmax(Z[j][s_I..., :])[2] for s_I in a1]
        df = DataFrame(a1 = a1, a2 = a2)
        pretty_table(df, ["$((I_j[j]...,))", "$j"])
    end
end

"""Print utility distribution

# Examples
```julia
udist = UtilityDistribution(G, X, Z)
print_utility_distribution(udist)
```
"""
function print_utility_distribution(udist::UtilityDistribution; util_fmt="%f", prob_fmt="%f")
    @unpack u, p = udist
    df = DataFrame(utility = u, probability = p)
    formatters = (
        ft_printf(util_fmt, [1]),
        ft_printf(prob_fmt, [2]))
    pretty_table(df, row_names=1:length(u); formatters = formatters)
end

"""Print state probabilities with fixed states.

# Examples
```julia
sprobs = StateProbabilities(G, X, Z)
print_state_probabilities(sprobs, G.C)
print_state_probabilities(sprobs, G.D)
```
"""
function print_state_probabilities(sprobs::StateProbabilities, nodes::Vector{Node}; prob_fmt="%f")
    probs = sprobs.probs
    fixed = sprobs.fixed

    prob(p, state) = if 1≤state≤length(p) p[state] else NaN end
    fix_state(i) = if i∈keys(fixed) string(fixed[i]) else "" end

    # Maximum number of states
    limit = maximum(length(probs[i]) for i in nodes)
    states = 1:limit
    df = DataFrame()
    for state in states
        df[!, Symbol(state)] = [prob(probs[i], state) for i in nodes]
    end
    df[!, :fixed] = [fix_state(i) for i in nodes]
    pretty_table(df, row_names=nodes; formatters = ft_printf(prob_fmt, states))
end

"""Value-at-risk."""
function _value_at_risk(u, p, α)
    cs = cumsum(p[sortperm(u)])
    VaR = u[findfirst(x -> x>α, cs)]
    return if isnothing(VaR) 0.0 else VaR end
end

"""Conditional value-at-risk."""
function _conditional_value_at_risk(u, p, α)
    x_α = _value_at_risk(u, p, α)
    tail = u .≤ x_α
    return (sum(u[tail] .* p[tail]) - (sum(p[tail]) - α) * x_α) / α
end

"""Print statistics."""
function print_statistics(udist::UtilityDistribution; fmt = "%f")
    @unpack u, p = udist
    w = ProbabilityWeights(p)
    pretty_table(
        [mean(u, w), std(u, w, corrected=false), skewness(u, w), kurtosis(u, w)],
        ["Statistics"],
        row_names = ["Mean", "Std", "Skewness", "Kurtosis"],
        formatters = ft_printf(fmt))
end

"""Print risk measures."""
function print_risk_measures(udist::UtilityDistribution, αs::Vector{Float64}; fmt = "%f")
    @unpack u, p = udist
    VaR = [_value_at_risk(u, p, α) for α in αs]
    CVaR = [_conditional_value_at_risk(u, p, α) for α in αs]
    pretty_table(
        hcat(αs, VaR, CVaR),
        ["α", "VaR_α", "CVaR_α"],
        formatters = ft_printf(fmt))
end
