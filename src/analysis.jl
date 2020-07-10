using Printf, Parameters

"""Test is path is compatible with a decision strategy."""
function is_compatible(s, z, D, I_j)
    all(isone(z[j][s[[I_j[j]; j]]...]) for j in D)
end

"""Generate all active paths from a decision strategy with fixed states."""
function active_paths(z, diagram::InfluenceDiagram, fixed::Dict{Int, Int})
    @unpack D, S_j, I_j = diagram
    return (s for s in paths(S_j, fixed) if is_compatible(s, z, D, I_j))
end

"""Generate all active paths from a decision strategy."""
function active_paths(z, diagram::InfluenceDiagram)
    active_paths(z, diagram, Dict{Int, Int}())
end

"""State probabilities."""
function state_probabilities(z, diagram::InfluenceDiagram, params::Params, prior::Float64, fixed::Dict{Int, Int})::Dict{Int, Vector{Float64}}
    @unpack C, D, S_j, I_j = diagram
    @unpack X = params
    probs = Dict(i => zeros(S_j[i]) for i in (C ∪ D))
    for s in active_paths(z, diagram, fixed), i in (C ∪ D)
        probs[i][s[i]] += path_probability(s, C, I_j, X) / prior
    end
    return probs
end

"""State probabilities."""
function state_probabilities(z, diagram::InfluenceDiagram, params::Params)
    return state_probabilities(z, diagram, params, 1.0, Dict{Int, Int}())
end

"""The probability distribution of utilities.

```julia
using Plots
x, y = distribution(z, diagram, params)
y2 = cumsum(y)
p = plot(x, y,
    linestyle=:dash,
    markershape=:circle,
    ylims=(0, 1.1),
    label="Distribution",
    legend=:topleft)
plot!(p, x, y2,
    linestyle=:dash,
    markershape=:circle,
    label="Cumulative distribution")
savefig(p, "distribution.svg")
```
"""
function utility_distribution(z, diagram::InfluenceDiagram, params::Params)
    @unpack C, D, V, I_j, S_j = diagram
    @unpack X, Y = params
    utilities = Vector{Float64}()
    probabilities = Vector{Float64}()
    for s in active_paths(z, diagram)
        push!(utilities, path_utility(s, Y, I_j, V))
        push!(probabilities, path_probability(s, C, I_j, X))
    end
    i = sortperm(utilities[:])
    x = utilities[i]
    y = probabilities[i]

    # Squash equal consecutive utilities into one, sum probabilities
    j = 1
    x2 = [x[1]]
    y2 = [y[1]]
    for k in 2:length(x)
        if x[k] == x2[j]
            y2[j] += y[k]
        else
            push!(x2, x[k])
            push!(y2, y[k])
            j += 1
        end
    end

    return x2, y2
end

"""Print number of paths, number of active paths and expected utility."""
function print_results(z, diagram, params)
    @unpack C, D, V, I_j, S_j = diagram
    @unpack X, Y = params
    # Total expected utility of the decision strategy.
    expected_utility = sum(
        path_probability(s, C, I_j, X) * path_utility(s, Y, I_j, V)
        for s in active_paths(z, diagram))
    println("Number of paths: ", prod(S_j))
    println("Number of active paths: ", prod(S_j[j] for j in C))
    println("Expected utility: ", expected_utility)
end

"""Print decision strategy."""
function print_decision_strategy(z, diagram)
    @unpack C, D, V, I_j, S_j = diagram
    println("j | s_I(j) | s_j")
    for j in D
        println("I($j) = $(I_j[j])")
        for s_I in paths(S_j[I_j[j]])
            _, s_j = findmax(z[j][s_I..., :])
            @printf("%i | %s | %s \n", j, s_I, s_j)
        end
    end
end

"""Print state probabilities with fixed states."""
function print_state_probabilities(probs, nodes, states, fixed::Dict{Int, Int})
    print("Node")
    for label in states
        print(" | ", label)
    end
    println()
    for i in nodes
        @printf("%4i", i)
        for prob in probs[i]
            @printf(" | %0.3f", prob)
        end
        if i ∈ keys(fixed)
            @printf(" | Fixed to state %i", fixed[i])
        end
        println()
    end
end

"""Print state probabilities."""
function print_state_probabilities(probs, nodes, labels)
    return print_state_probabilities(probs, nodes, labels, Dict{Int, Int}())
end
