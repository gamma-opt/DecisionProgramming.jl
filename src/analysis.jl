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
    ActivePaths(G, Z, fixed) = !all(k∈G.C for k in keys(fixed)) ? error("You can only fix chance states.") : new(G, Z, fixed)
end

function ActivePaths(G::InfluenceDiagram, Z::DecisionStrategy)
    ActivePaths(G, Z, Dict{Int, Int}())
end

"""Decision state Z_j : s_I(j) → s_j"""
function decision_state(G::InfluenceDiagram, Z::DecisionStrategy, s::Path, j::Int)
    findmax(Z[j][s[G.I_j[j]]..., :])[2]
end

function active_path(G::InfluenceDiagram, Z::DecisionStrategy, s_C::Path)
    @unpack C, D, I_j = G
    n = length(C) + length(D)
    s = Array{Int}(undef, n)
    s[C] .= s_C
    for j in D
        s[j] = decision_state(G, Z, (s...,), j)
    end
    return (s...,)
end

function Base.iterate(a::ActivePaths)
    @unpack G, Z = a
    @unpack C, S_j = G
    ks = sort(collect(keys(a.fixed)))
    fixed = Dict{Int, Int}(i => a.fixed[k] for (i, k) in enumerate(ks))
    iter = paths(S_j[C], fixed)
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

Base.eltype(::Type{ActivePaths}) = Path
Base.length(a::ActivePaths) = prod(a.G.S_j[a.G.C])

"""The probability mass function."""
function utility_distribution(Z::DecisionStrategy, G::InfluenceDiagram, X::Probabilities, U::UtilityFunction)
    @unpack C, D, V, I_j, S_j = G

    # Extract utilities and probabilities of active paths
    S_Z = ActivePaths(G, Z)
    utilities = Vector{Float64}(undef, length(S_Z))
    probabilities = Vector{Float64}(undef, length(S_Z))
    for (i, s) in enumerate(S_Z)
        utilities[i] = U(s)
        probabilities[i] = path_probability(s, G, X)
    end

    # Sort by utilities
    i = sortperm(utilities)
    u = utilities[i]
    p = probabilities[i]

    # Compute the probability mass function
    u2 = unique(u)
    p2 = similar(u2)
    j = 1
    p2[j] = p[1]
    for k in 2:length(u)
        if u[k] == u2[j]
            p2[j] += p[k]
        else
            j += 1
            p2[j] = p[k]
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
