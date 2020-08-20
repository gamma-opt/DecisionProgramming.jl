using Random, Plots
using DecisionProgramming

scale(x::Float64, low::Float64, high::Float64) = x * (high - low) + low

function random_probability_distribution(rng::AbstractRNG, n_neg::Int, n_pos::Int, low::Real, high::Real)
    low < high || error("")
    u_neg = rand(rng, n_neg)
    u_neg = scale.(u_neg, low, 0.0)
    u_pos = rand(rng, n_pos)
    u_pos = scale.(u_pos, 0.0, high)
    u = [u_neg; u_pos]
    p = rand(rng, n_neg + n_pos)
    p = p / sum(p)
    i = sortperm(u)
    return u[i], p[i]
end

hair(plt, x, y) = plot!(plt, [x, x], [0, y], linewidth=3, color=:grey, alpha=0.5, label=false)

function plot_distribution(x, y, x_α; plt=plot())
    for (x2, y2) in zip(x, y)
        hair(plt, x2, y2)
    end
    tail = x .≤ x_α
    head = x .> x_α
    plot!(plt, ylims = (0, 1.15 * maximum(y)))
    plot!(plt, x[tail], y[tail], linewidth=0, markershape=:circle, markercolor=:darkred,
          label="Tail")
    plot!(plt, x[head], y[head], linewidth=0, markershape=:circle, markercolor=:lightblue,
          label="Head")
    return plt
end

function plot_risk_measures(u, p, α)
    mean = sum(u.*p)
    VaR = value_at_risk(u, p, α)
    CVaR = conditional_value_at_risk(u, p, α)

    plt1 = plot(
        title="Probability distribution",
        legend=false,
        xlabel="x",
        ylabel="f(x)"
    )
    plot_distribution(u, p, VaR; plt=plt1)
    plot!(plt1, [mean], [0],
        linewidth=0, markershape=:diamond,
        markersize=6, markercolor=:green,
        label="E")
    plot!(plt1, [VaR], [0],
        linewidth=0, markershape=:diamond,
        markersize=6, markercolor=:darkblue,
        label="VaR")
    plot!(plt1, [CVaR], [0],
        linewidth=0, markershape=:diamond,
        markersize=6, markercolor=:darkorange,
        label="CVaR")

    plt2 = plot(
        title="Cumulative distribution",
        xlabel="x",
        ylabel="F(x)"
    )
    plot_distribution(u, cumsum(p), VaR; plt=plt2)
    plot!(plt2, u, [α for _ in u],
          legend=false, linewidth=2, label="α", linecolor=:darkorange)
    plot!(plt2, [VaR], [0],
        linewidth=0, markershape=:diamond,
        markersize=6, markercolor=:darkblue,
        label="VaR")

    plt = plot(plt1, plt2, layout=(2, 1), legend=:outerright, size=(720, 400))
end

# rng = MersenneTwister(1)
# u, p = random_probability_distribution(rng, 4, 8, -2.0, 3.0)
# α = 0.25
# plt = plot_risk_measures(u, p, α)
# savefig(plt, "../../docs/src/decision-programming/figures/risk_measures.svg")
# savefig(plt, "risk_measures.svg")
