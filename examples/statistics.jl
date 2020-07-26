using StatsBase
using StatsBase.Statistics

"""Value-at-risk."""
function _value_at_risk(u, p, α)
    u_α = u[p .≤ α]
    return if isempty(u_α) 0.0 else -maximum(u_α) end
end

"""Conditional value-at-risk."""
function _conditional_value_at_risk(u, p, α)
    x_α = -_value_at_risk(u, p, α)
    tail = u .≤ x_α
    return -(sum(u[tail] .* p[tail])/sum(p[tail]) + (α - sum(p[tail])) * x_α) / α
end

function print_stats(u, p; αs=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0])
    w = ProbabilityWeights(p)
    println("Mean: ", mean(u, w))
    println("Std: ", std(u, w, corrected=false))
    println("Skewness: ", skewness(u, w))
    println("Kurtosis: ", kurtosis(u, w))
    println()
    println("  α | VaR_α(Z) | CVaR_α(Z)")
    for α in αs
        @printf("%.3f | %.2f | %.2f \n", α,
            _value_at_risk(u, p, α),
            _conditional_value_at_risk(u, p, α))
    end
end
