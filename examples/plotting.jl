using Dates
using Plots


function create_directory(name)
    directory = joinpath(name, string(now()))
    if !ispath(directory)
        mkpath(directory)
    end
    return directory
end

hair(plt, x, y) = plot!(plt, [x, x], [0, y], linewidth=3, color=:grey, alpha=0.5, label=false)

function plot_distribution(u, p; plt=plot(), label="")
    plot!(plt, u, p,
        linewidth=0,
        markershape=:circle,
        ylims=(0, 1.15 * maximum(p)),
        xticks = LinRange(minimum(u), maximum(u), 10),
        yticks = LinRange(0, maximum(p), 11),
        label=label,
        legend=:topleft)
    for (x, y) in zip(u, p)
        hair(plt, x, y)
    end
    return plt
end

function plot_cumulative_distribution(u, p; plt=plot(), label="")
    plot!(plt, u, cumsum(p),
        linewidth=0,
        markershape=:circle,
        ylims=(0, 1.15 * 1),
        xticks = LinRange(minimum(u), maximum(u), 10),
        yticks = 0:0.1:1,
        label=label,
        legend=:topleft)
    for (x, y) in zip(u, cumsum(p))
        hair(plt, x, y)
    end
    return plt
end
