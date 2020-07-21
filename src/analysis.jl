using Parameters

"""Interface for iterating over active paths given influence diagram and decision strategy.

```julia
for s in ActivePaths(G, Z)
    ...
end
```
"""
struct ActivePaths
    G::InfluenceDiagram
    Z::DecisionStrategy
    fixed::Dict{Int, Int}
end

ActivePaths(G, Z) = ActivePaths(G, Z, Dict{Int, Int}())

function active_path(G::InfluenceDiagram, Z::DecisionStrategy, s_C::NTuple{N, Int}) where N
    @unpack C, D, I_j = G
    n = length(C) + length(D)
    s = Array{Int}(undef, n)
    s[C] .= s_C
    for j in D
        _, s[j] = findmax(Z[j][s[I_j[j]]..., :])
    end
    return (s...,)
end

function Base.iterate(a::ActivePaths)
    @unpack G, Z = a
    iter = paths(G.S_j[G.C], a.fixed)
    next = iterate(iter)
    if next !== nothing
        s_C, state = next
        return (active_path(G, Z, s_C), (iter, state))
    end
end

function Base.iterate(a::ActivePaths, gen)
    @unpack G, Z = a
    iter, state = gen
    next = iterate(iter, state)
    if next !== nothing
        s_C, state = next
        return (active_path(G, Z, s_C), (iter, state))
    end
end

Base.eltype(::Type{ActivePaths}) = NTuple{N, Int} where N

"""The probability mass function."""
function utility_distribution(Z::DecisionStrategy, G::InfluenceDiagram, X::Probabilities, U::UtilityFunction)
    @unpack C, D, V, I_j, S_j = G
    utilities = Vector{Float64}()
    probabilities = Vector{Float64}()
    for s in ActivePaths(G, Z)
        push!(utilities, U(s))
        push!(probabilities, path_probability(s, G, X))
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
function state_probabilities(Z::DecisionStrategy, G::InfluenceDiagram, X::Probabilities, prior::Float64, fixed::Dict{Int, Int})::Dict{Int, Vector{Float64}}
    @unpack C, D, S_j, I_j = G
    probs = Dict(i => zeros(S_j[i]) for i in (C ∪ D))
    for s in ActivePaths(G, Z, fixed), i in (C ∪ D)
        probs[i][s[i]] += path_probability(s, G, X) / prior
    end
    return probs
end

"""State probabilities."""
function state_probabilities(Z::DecisionStrategy, G::InfluenceDiagram, X::Probabilities)
    return state_probabilities(Z, G, X, 1.0, Dict{Int, Int}())
end
