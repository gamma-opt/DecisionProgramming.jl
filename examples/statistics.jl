using StatsBase
using StatsBase.Statistics

"""Value-at-risk."""
function VaR(u, w, α)
    u_α = u[w .≤ α]
    return if isempty(u_α) 0.0 else -maximum(u_α) end
end

"""Conditional value-at-risk."""
function CVaR(u, w, α)
    x_α = -VaR(u, w, α)
    tail = u .≤ x_α
    return -(mean(u[tail], w[tail]) + (α - sum(w[tail])) * x_α) / α
end

function print_stats(u, p; αs=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0])
    w = ProbabilityWeights(p)
    println("Mean: ", mean(u, w))
    println("Std: ", std(u, w, corrected=false))
    println("Skewness: ", skewness(u, w))
    println("Kurtosis: ", kurtosis(u, w))
    println("  α | VaR_α(Z) | CVaR_α(Z)")
    for α in αs
        @printf("%.3f | %.2f | %.2f \n", α, VaR(u, w, α), CVaR(u, w, α))
    end
end
