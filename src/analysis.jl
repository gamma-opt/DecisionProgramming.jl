using Parameters

"""Interface for iterating over active paths given influence diagram and decision strategy.

1) Initialize path `s` of length `n`
2) Fill chance states `s[C]` by generating subpaths `paths(G.C)`
3) Fill decision states `s[D]` by decision strategy `Z` and path `s`

# Examples
```julia
for s in ActivePaths(G, Z)
    ...
end
```
"""
struct ActivePaths
    G::InfluenceDiagram
    Z::DecisionStrategy
    fixed::Dict{Node, State}
    function ActivePaths(G, Z, fixed)
        !all(k∈G.C for k in keys(fixed)) && error("You can only fix chance states.")
        new(G, Z, fixed)
    end
end

function ActivePaths(G::InfluenceDiagram, Z::DecisionStrategy)
    ActivePaths(G, Z, Dict{Node, State}())
end

function active_path(G::InfluenceDiagram, Z::DecisionStrategy, s_C::Path)
    @unpack C, D, I_j = G
    n = length(C) + length(D)
    s = Array{Int}(undef, n)
    s[C] .= s_C
    for j in D
        s[j] = Z(j, (s...,), G)
    end
    return (s...,)
end

function Base.iterate(a::ActivePaths)
    @unpack G, Z = a
    @unpack C, S_j = G
    if isempty(a.fixed)
        iter = paths(S_j[C])
    else
        ks = sort(collect(keys(a.fixed)))
        fixed = Dict{Int, Int}(i => a.fixed[k] for (i, k) in enumerate(ks))
        iter = paths(S_j[C], fixed)
    end
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

"""UtilityDistribution type."""
struct UtilityDistribution
    u::Vector{Float64}
    p::Vector{Float64}
end

"""Constructs the probability mass function for path utilities on active paths.

# Examples
```julia
UtilityDistribution(G, P, U, Z)
```
"""
function UtilityDistribution(G::InfluenceDiagram, P::PathProbability, U::PathUtility, Z::DecisionStrategy)
    @unpack C, D, V, I_j, S_j = G

    # Extract utilities and probabilities of active paths
    S_Z = ActivePaths(G, Z)
    utilities = Vector{Float64}(undef, length(S_Z))
    probabilities = Vector{Float64}(undef, length(S_Z))
    for (i, s) in enumerate(S_Z)
        utilities[i] = U(s)
        probabilities[i] = P(s)
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

    UtilityDistribution(u2, p2)
end

"""StateProbabilities type."""
struct StateProbabilities
    probs::Dict{Node, Vector{Float64}}
    fixed::Dict{Node, State}
end

"""Associates each node with array of conditional probabilities for each of its states occuring in active paths given fixed states and prior probability.

# Examples
```julia
# Prior probabilities
prev = StateProbabilities(G, P, Z)

# Select node and fix its state
node = 1
state = 2
StateProbabilities(G, P, Z, node, state, prev)
```
"""
function StateProbabilities(G::InfluenceDiagram, P::PathProbability, Z::DecisionStrategy, node::Node, state::State, prev::StateProbabilities)
    @unpack C, D, S_j, I_j = G
    prior = prev.probs[node][state]
    fixed = prev.fixed
    push!(fixed, node => state)
    probs = Dict(i => zeros(S_j[i]) for i in (C ∪ D))
    for s in ActivePaths(G, Z, fixed), i in (C ∪ D)
        probs[i][s[i]] += P(s) / prior
    end
    StateProbabilities(probs, fixed)
end

"""Associates each node with array of probabilities for each of its states occuring in active paths.

# Examples
```julia
StateProbabilities(G, P, Z)
```
"""
function StateProbabilities(G::InfluenceDiagram, P::PathProbability, Z::DecisionStrategy)
    @unpack C, D, S_j, I_j = G
    probs = Dict(i => zeros(S_j[i]) for i in (C ∪ D))
    for s in ActivePaths(G, Z), i in (C ∪ D)
        probs[i][s[i]] += P(s)
    end
    StateProbabilities(probs, Dict{Node, State}())
end
