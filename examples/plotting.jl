using Dates
using Plots


function create_directory(dirpath)
    directory = joinpath(dirpath, string(now()))
    if !ispath(directory)
        mkpath(directory)
    end
    return directory
end

hair(plt, x, y) = plot!(plt, [x, x], [0, y], linewidth=3, color=:grey, alpha=0.5, label=false)

"""Plot probability mass function.

# Examples
```julia
# Probability mass function
plot_distribution(u, p)
# Cumulative distribution function
plot_distribution(u, cumsum(p))
```
"""
function plot_distribution(u, p; plt=plot(), label="")
    x = 1:length(u)
    y = p
    plot!(plt, x, y,
        linewidth=0,
        markershape=:circle,
        ylims=(0, 1.15 * maximum(p)),
        xformatter = tick -> "$(u[Int(tick)])",
        yticks = LinRange(0, maximum(p), 11),
        label=label,
        legend=:topleft)
    for (x2, y2) in zip(x, y)
        hair(plt, x2, y2)
    end
    return plt
end
