using Printf, Parameters, DataFrames, PrettyTables
using StatsBase, StatsBase.Statistics

"""Print decision strategy.

# Examples
```julia
print_decision_strategy(S, Z)
```
"""
function print_decision_strategy(S::States, Z::GlobalDecisionStrategy)
    for (d, Z_j) in zip(Z.D, Z.Z_j)
        a1 = collect(paths(S[d.I_j]))[:]
        a2 = [Z_j(s_I) for s_I in a1]
        labels = fill("States", length(a1))
        df = DataFrame(labels = labels, a1 = a1, a2 = a2)
        pretty_table(df, ["Nodes", "$((d.I_j...,))", "$(d.j)"])
    end
end

"""Print utility distribution

# Examples
```julia
udist = UtilityDistribution(S, P, U, Z)
print_utility_distribution(udist)
```
"""
function print_utility_distribution(udist::UtilityDistribution; util_fmt="%f", prob_fmt="%f")
    @unpack u, p = udist
    df = DataFrame(Utility = u, Probability = p)
    filter!(row -> row[:Probability] > 0, df)
    formatters = (
        ft_printf(util_fmt, [1]),
        ft_printf(prob_fmt, [2]))
    pretty_table(df; formatters = formatters)
end

"""Print state probabilities with fixed states.

# Examples
```julia
sprobs = StateProbabilities(S, P, U, Z)
print_state_probabilities(sprobs, [c.j for c in C])
print_state_probabilities(sprobs, [d.j for d in D])
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
    df[!, :Node] = nodes
    for state in states
        df[!, Symbol("State $state")] = [prob(probs[i], state) for i in nodes]
    end
    df[!, Symbol("Fixed state")] = [fix_state(i) for i in nodes]
    pretty_table(df; formatters = ft_printf(prob_fmt, (first(states)+1):(last(states)+1)))
end

"""Value-at-risk."""
function value_at_risk(u::Vector{Float64}, p::Vector{Float64}, α::Float64)
    cs = cumsum(p[sortperm(u)])
    index = findfirst(x -> x>α, cs)
    return if isnothing(index) 0.0 else u[index] end
end

"""Conditional value-at-risk."""
function conditional_value_at_risk(u::Vector{Float64}, p::Vector{Float64}, α::Float64)
    x_α = value_at_risk(u, p, α)
    tail = u .≤ x_α
    return (sum(u[tail] .* p[tail]) - (sum(p[tail]) - α) * x_α) / α
end

"""Print statistics."""
function print_statistics(udist::UtilityDistribution; fmt = "%f")
    @unpack u, p = udist
    w = ProbabilityWeights(p)
    names = ["Mean", "Std", "Skewness", "Kurtosis"]
    statistics = [mean(u, w), std(u, w, corrected=false), skewness(u, w), kurtosis(u, w)]
    df = DataFrame(Name = names, Statistics = statistics)
    pretty_table(df, formatters = ft_printf(fmt, [2]))
end

"""Print risk measures."""
function print_risk_measures(udist::UtilityDistribution, αs::Vector{Float64}; fmt = "%f")
    @unpack u, p = udist
    VaR = [value_at_risk(u, p, α) for α in αs]
    CVaR = [conditional_value_at_risk(u, p, α) for α in αs]
    df = DataFrame(α = αs, VaR = VaR, CVaR = CVaR)
    pretty_table(df, formatters = ft_printf(fmt))
end
