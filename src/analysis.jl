using Parameters

"""Interface for iterating over active paths given influence diagram and decision strategy.

1) Initialize path `s` of length `n`
2) Fill chance states `s[C]` by generating subpaths `paths(G.C)`
3) Fill decision states `s[D]` by decision strategy `Z` and path `s`

# Examples
```julia
for s in ActivePaths(S, C, Z)
    ...
end
```
"""
struct ActivePaths
    S::States
    C::Vector{ChanceNode}
    Z::GlobalDecisionStrategy
    fixed::Dict{Node, State}
    function ActivePaths(S, C, Z, fixed)
        C_j = Set([c.j for c in C])
        !all(k∈C_j for k in keys(fixed)) && error("You can only fix chance states.")
        new(S, C, Z, fixed)
    end
end

function ActivePaths(S::States, C::Vector{ChanceNode}, Z::GlobalDecisionStrategy)
    ActivePaths(S, C, Z, Dict{Node, State}())
end

function active_path(S::States, C::Vector{ChanceNode}, Z::GlobalDecisionStrategy, s_C::Path)
    s = Array{Int}(undef, length(S))
    for (c, s_C_j) in zip(C, s_C)
        s[c.j] = s_C_j
    end
    for (d, Z_j) in zip(Z.D, Z.Z_j)
        s[d.j] = Z_j((s...,), d.I_j)
    end
    return (s...,)
end

function Base.iterate(a::ActivePaths)
    @unpack S, C, Z = a
    C_j = [c.j for c in C]
    if isempty(a.fixed)
        iter = paths(S[C_j])
    else
        ks = sort(collect(keys(a.fixed)))
        fixed = Dict{Int, Int}(i => a.fixed[k] for (i, k) in enumerate(ks))
        iter = paths(S[C_j], fixed)
    end
    next = iterate(iter)
    if next !== nothing
        s_C, state = next
        return (active_path(S, C, Z, s_C), (iter, state))
    end
end

function Base.iterate(a::ActivePaths, gen)
    @unpack S, C, Z = a
    iter, state = gen
    next = iterate(iter, state)
    if next !== nothing
        s_C, state = next
        return (active_path(S, C, Z, s_C), (iter, state))
    end
end

Base.eltype(::Type{ActivePaths}) = Path
Base.length(a::ActivePaths) = prod(a.S[c.j] for c in a.C)

"""UtilityDistribution type."""
struct UtilityDistribution
    u::Vector{Float64}
    p::Vector{Float64}
end

"""Constructs the probability mass function for path utilities on active paths.

# Examples
```julia
UtilityDistribution(S, P, U, Z)
```
"""
function UtilityDistribution(S::States, P::PathProbability, U::AbstractPathUtility, Z::GlobalDecisionStrategy)
    # Extract utilities and probabilities of active paths
    S_Z = ActivePaths(S, P.C, Z)
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
StateProbabilities(S, P, Z, node, state, prev)
```
"""
function StateProbabilities(S::States, P::PathProbability, Z::GlobalDecisionStrategy, node::Node, state::State, prev::StateProbabilities)
    prior = prev.probs[node][state]
    fixed = prev.fixed
    push!(fixed, node => state)
    probs = Dict(i => zeros(S[i]) for i in 1:length(S))
    for s in ActivePaths(S, P.C, Z, fixed), i in 1:length(S)
        probs[i][s[i]] += P(s) / prior
    end
    StateProbabilities(probs, fixed)
end

"""Associates each node with array of probabilities for each of its states occuring in active paths.

# Examples
```julia
StateProbabilities(S, P, Z)
```
"""
function StateProbabilities(S::States, P::PathProbability, Z::GlobalDecisionStrategy)
    probs = Dict(i => zeros(S[i]) for i in 1:length(S))
    for s in ActivePaths(S, P.C, Z), i in 1:length(S)
        probs[i][s[i]] += P(s)
    end
    StateProbabilities(probs, Dict{Node, State}())
end
