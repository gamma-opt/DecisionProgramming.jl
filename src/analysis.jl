using Parameters

"""Test is path is compatible with a decision strategy."""
function is_compatible(s, z, D, I_j)
    all(isone(z[j][s[[I_j[j]; j]]...]) for j in D)
end

"""Generate all active paths from a decision strategy with fixed states."""
function active_paths(z::DecisionStrategy, diagram::InfluenceDiagram, fixed::Dict{Int, Int})
    @unpack D, S_j, I_j = diagram
    return (s for s in paths(S_j, fixed) if is_compatible(s, z, D, I_j))
end

"""Generate all active paths from a decision strategy."""
function active_paths(z::DecisionStrategy, diagram::InfluenceDiagram)
    active_paths(z, diagram, Dict{Int, Int}())
end

"""The probability mass function."""
function utility_distribution(z::DecisionStrategy, diagram::InfluenceDiagram, params::Params, U::UtilityFunction)
    @unpack C, D, V, I_j, S_j = diagram
    @unpack X, Y = params
    utilities = Vector{Float64}()
    probabilities = Vector{Float64}()
    for s in active_paths(z, diagram)
        push!(utilities, U(s))
        push!(probabilities, path_probability(s, diagram, params))
    end
    i = sortperm(utilities[:])
    u = utilities[i]
    p = probabilities[i]

    # Squash equal consecutive utilities into one, sum probabilities
    j = 1
    u2 = [u[1]]
    p2 = [p[1]]
    for k in 2:length(u)
        if u[k] == u2[j]
            p2[j] += p[k]
        else
            push!(u2, u[k])
            push!(p2, p[k])
            j += 1
        end
    end

    return u2, p2
end

"""State probabilities."""
function state_probabilities(z::DecisionStrategy, diagram::InfluenceDiagram, params::Params, prior::Float64, fixed::Dict{Int, Int})::Dict{Int, Vector{Float64}}
    @unpack C, D, S_j, I_j = diagram
    probs = Dict(i => zeros(S_j[i]) for i in (C ∪ D))
    for s in active_paths(z, diagram, fixed), i in (C ∪ D)
        probs[i][s[i]] += path_probability(s, diagram, params) / prior
    end
    return probs
end

"""State probabilities."""
function state_probabilities(z::DecisionStrategy, diagram::InfluenceDiagram, params::Params)
    return state_probabilities(z, diagram, params, 1.0, Dict{Int, Int}())
end
